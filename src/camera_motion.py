"""
Camera Ingestion, Optical Flow Motion Estimation, and Stump Calibration Module
Part of the str8bat AI Engineer Technical Take-Home Assignment.

This module performs:
1. Video frame extraction with cleanly burned frame index and timestamp overlays.
2. Foreground masking of dynamic actors (batter, keeper, umpire, ball, bat) via YOLO.
3. Feature tracking and partial affine RANSAC estimation for camera ego-motion (pan/tilt/zoom).
4. Striker stump regulation measurement and pitch-plane metric scale factor calibration.
5. Export of complete camera metrics to JSON.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO


REGULATION_STUMP_HEIGHT_M = 0.711  # ICC standard 28 inches = 0.7112 m
STRIKER_STUMP_TOP_PX = [340, 809]   # [x, y] coordinates of top of bails/stumps
STRIKER_STUMP_BOT_PX = [333, 908]   # [x, y] coordinates of base of stumps on pitch


def inspect_video(video_path: str) -> Dict[str, Any]:
    """Inspects video container and reads actual decodable frames."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Count decodable frames
    total_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        total_frames += 1
    cap.release()

    if fps <= 0:
        fps = 30.0

    duration_seconds = round(total_frames / fps, 3)

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "total_frames": total_frames,
        "duration_seconds": duration_seconds
    }


def extract_and_overlay_frames(video_path: str, output_dir: str) -> List[np.ndarray]:
    """
    Extracts all frames into output_dir as frame_%04d.png with frame index
    and timestamp cleanly burned into the top-left corner.
    Returns list of original raw frames (BGR).
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    raw_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frames.append(frame.copy())

        # Create overlay copy
        overlay_frame = frame.copy()
        timestamp = frame_idx / fps
        label = f"F: {frame_idx:02d} | T: {timestamp:.3f}s"

        # Clean background pill for readability
        pad = 8
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        box_tl = (14, 14)
        box_br = (14 + text_w + pad * 2, 14 + text_h + pad * 2)

        # Semi-transparent dark pill background
        sub_img = overlay_frame[box_tl[1]:box_br[1], box_tl[0]:box_br[0]]
        black_rect = np.zeros_like(sub_img)
        cv2.addWeighted(black_rect, 0.7, sub_img, 0.3, 0, sub_img)
        cv2.rectangle(overlay_frame, box_tl, box_br, (50, 50, 50), 1)

        # Draw anti-aliased text
        text_pos = (box_tl[0] + pad, box_br[1] - pad - 2)
        cv2.putText(
            overlay_frame,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, overlay_frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted and saved {len(raw_frames)} frames with overlays to {output_dir}")
    return raw_frames


def create_foreground_mask(frame: np.ndarray, model: YOLO, pad_px: int = 15) -> np.ndarray:
    """
    Detects dynamic foreground objects (persons, bats, balls, rackets)
    and produces a binary mask (255 = static background/pitch, 0 = foreground).
    """
    H, W = frame.shape[:2]
    mask = np.ones((H, W), dtype=np.uint8) * 255

    # Run YOLO detection
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        # Classes: 0: person, 32: sports ball, 34: baseball bat, 38: tennis racket
        if cls_id in [0, 32, 34, 38]:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 = max(0, x1 - pad_px)
            y1 = max(0, y1 - pad_px)
            x2 = min(W, x2 + pad_px)
            y2 = min(H, y2 + pad_px)
            mask[y1:y2, x1:x2] = 0

    return mask


def estimate_camera_motion(
    raw_frames: List[np.ndarray],
    fps: float,
    model: Optional[YOLO] = None
) -> List[Dict[str, Any]]:
    """
    Calculates camera displacement (dx_cam, dy_cam), scale factor scale_cam,
    and camera speed v_cam_px_s for every frame t from 0 to N-1.
    Frame 0 displacement is (0, 0).
    """
    if model is None:
        model = YOLO("yolov8n.pt")

    n_frames = len(raw_frames)
    motions: List[Dict[str, Any]] = []

    # Frame 0 baseline
    motions.append({
        "frame_idx": 0,
        "timestamp": 0.0,
        "dx_cam": 0.0,
        "dy_cam": 0.0,
        "scale_cam": 1.0,
        "v_cam_px_s": 0.0,
        "mean_frame_diff": 0.0,
        "is_duplicate_frame": False
    })

    for t in range(1, n_frames):
        prev = raw_frames[t - 1]
        curr = raw_frames[t]
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        H, W = prev_gray.shape
        timestamp = round(t / fps, 4)

        mean_frame_diff = float(np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float))))
        is_duplicate = bool(mean_frame_diff < 1.5)

        # Detect shot cuts / abrupt transitions (e.g. sky to pitch cut at frame 21->22)
        if mean_frame_diff > 40.0:
            motions.append({
                "frame_idx": t,
                "timestamp": timestamp,
                "dx_cam": 0.0,
                "dy_cam": 0.0,
                "scale_cam": 1.0,
                "v_cam_px_s": 0.0,
                "mean_frame_diff": round(mean_frame_diff, 3),
                "is_duplicate_frame": False
            })
            continue

        # Handle duplicate broadcast pulldown frames
        if is_duplicate:
            motions.append({
                "frame_idx": t,
                "timestamp": timestamp,
                "dx_cam": 0.0,
                "dy_cam": 0.0,
                "scale_cam": 1.0,
                "v_cam_px_s": 0.0,
                "mean_frame_diff": round(mean_frame_diff, 3),
                "is_duplicate_frame": True
            })
            continue

        # Mask out dynamic foreground objects
        mask = create_foreground_mask(prev, model, pad_px=15)

        # Detect good features to track in static background/pitch regions
        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=mask,
            maxCorners=400,
            qualityLevel=0.01,
            minDistance=8
        )

        # Fallback if too few features found (e.g. sky frames)
        if p0 is None or len(p0) < 6:
            p0 = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=400,
                qualityLevel=0.005,
                minDistance=5
            )

        if p0 is not None and len(p0) >= 4:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            good_0 = p0[st == 1]
            good_1 = p1[st == 1]

            if len(good_0) >= 4:
                M, inliers = cv2.estimateAffinePartial2D(
                    good_0, good_1,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=2000
                )

                if M is not None and inliers is not None and np.sum(inliers) >= 4:
                    tx = float(M[0, 2])
                    ty = float(M[1, 2])
                    # Camera pan ego-motion is opposite to scene feature shift
                    dx_cam = -tx
                    dy_cam = -ty
                    scale_cam = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
                    v_cam_px_s = float(np.sqrt(dx_cam ** 2 + dy_cam ** 2) * fps)
                else:
                    dx_cam, dy_cam, scale_cam, v_cam_px_s = 0.0, 0.0, 1.0, 0.0
            else:
                dx_cam, dy_cam, scale_cam, v_cam_px_s = 0.0, 0.0, 1.0, 0.0
        else:
            dx_cam, dy_cam, scale_cam, v_cam_px_s = 0.0, 0.0, 1.0, 0.0

        motions.append({
            "frame_idx": t,
            "timestamp": timestamp,
            "dx_cam": round(dx_cam, 3),
            "dy_cam": round(dy_cam, 3),
            "scale_cam": round(scale_cam, 4),
            "v_cam_px_s": round(v_cam_px_s, 2),
            "mean_frame_diff": round(mean_frame_diff, 3),
            "is_duplicate_frame": False
        })

    return motions


def calibrate_stumps() -> Dict[str, Any]:
    """
    Computes striker wicket stump calibration and pitch-plane metric scale factor.
    """
    top_x, top_y = STRIKER_STUMP_TOP_PX
    bot_x, bot_y = STRIKER_STUMP_BOT_PX
    height_px = float(np.sqrt((bot_x - top_x) ** 2 + (bot_y - top_y) ** 2))
    height_px = round(height_px, 2)

    meters_per_pixel = round(REGULATION_STUMP_HEIGHT_M / height_px, 6)

    return {
        "stump_top_px": STRIKER_STUMP_TOP_PX,
        "stump_bottom_px": STRIKER_STUMP_BOT_PX,
        "stump_height_px": height_px,
        "regulation_height_m": REGULATION_STUMP_HEIGHT_M,
        "meters_per_pixel": meters_per_pixel
    }


def compute_summary(motions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes motion statistics and finds peak pan frame."""
    speeds = [m["v_cam_px_s"] for m in motions]
    pan_displacements = [np.sqrt(m["dx_cam"] ** 2 + m["dy_cam"] ** 2) for m in motions]

    mean_speed = round(float(np.mean(speeds)), 2)
    max_speed = round(float(np.max(speeds)), 2)
    peak_pan_frame = int(np.argmax(pan_displacements))

    return {
        "mean_cam_speed_px_s": mean_speed,
        "max_cam_speed_px_s": max_speed,
        "peak_pan_frame": peak_pan_frame
    }


def run_pipeline(
    video_path: str = "input_video.mp4",
    frames_dir: str = "data/frames",
    output_json_path: str = "outputs/camera_metrics.json"
) -> Dict[str, Any]:
    """Runs the entire ingestion, calibration, and motion estimation pipeline."""
    print("=" * 60)
    print("STR8BAT TECHNICAL ASSIGNMENT - VIDEO INGESTION & CALIBRATION")
    print("=" * 60)

    # 1. Video Inspection
    print(f"\n[Step 1] Inspecting video: {video_path}")
    video_info = inspect_video(video_path)
    print(f"  Dimensions: {video_info['width']}x{video_info['height']}")
    print(f"  FPS:        {video_info['fps']}")
    print(f"  Frames:     {video_info['total_frames']}")
    print(f"  Duration:   {video_info['duration_seconds']}s")

    # 2. Frame Extraction & Burning Overlays
    print(f"\n[Step 2] Extracting frames with overlays to: {frames_dir}")
    raw_frames = extract_and_overlay_frames(video_path, frames_dir)

    # 3. Striker Stump Calibration
    print("\n[Step 3] Calibrating striker wicket stumps at pitch plane...")
    stump_calibration = calibrate_stumps()
    print(f"  Stump Top:            {stump_calibration['stump_top_px']}")
    print(f"  Stump Bottom:         {stump_calibration['stump_bottom_px']}")
    print(f"  Stump Height (px):    {stump_calibration['stump_height_px']} px")
    print(f"  Regulation Height:    {stump_calibration['regulation_height_m']} m")
    print(f"  Meters / Pixel:       {stump_calibration['meters_per_pixel']} m/px ({stump_calibration['meters_per_pixel']*1000:.3f} mm/px)")

    # 4. Foreground-Masked Optical Flow Camera Motion
    print("\n[Step 4] Tracking camera ego-motion with foreground masking...")
    yolo_model = YOLO("yolov8n.pt")
    frame_motions = estimate_camera_motion(raw_frames, video_info["fps"], yolo_model)
    summary = compute_summary(frame_motions)

    print(f"  Mean Camera Speed:    {summary['mean_cam_speed_px_s']} px/s")
    print(f"  Max Camera Speed:     {summary['max_cam_speed_px_s']} px/s")
    print(f"  Peak Pan Frame:       Frame {summary['peak_pan_frame']} (T: {frame_motions[summary['peak_pan_frame']]['timestamp']}s)")

    # 5. Export JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    payload = {
        "video_info": video_info,
        "stump_calibration": stump_calibration,
        "frame_motions": frame_motions,
        "summary": summary
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Step 5] Calibration results saved to: {output_json_path}")
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Ingestion, Camera Motion & Stump Calibration")
    parser.add_argument("--video", default="input_video.mp4", help="Path to input video")
    parser.add_argument("--frames-dir", "--frames_dir", dest="frames_dir", default="data/frames", help="Path to extracted frames directory")
    parser.add_argument("--output-json", "--output_json", dest="output_json", default="outputs/camera_metrics.json", help="Path to output JSON")

    args = parser.parse_args()
    run_pipeline(
        video_path=args.video,
        frames_dir=args.frames_dir,
        output_json_path=args.output_json
    )
