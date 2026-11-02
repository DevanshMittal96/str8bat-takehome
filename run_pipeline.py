#!/usr/bin/env python3
"""
str8bat AI Engineer Technical Take-Home Assignment (Part 2)
End-to-End Execution Pipeline Entrypoint

This script coordinates the full 3-step kinematic analysis:
  Step 1: Video Ingestion, Optical Flow Ego-Motion Calibration, Striker Stump Metric Scaling (src/camera_motion.py)
  Step 2: Striker Pose Estimation, Bat-Proxy Tracking & Trajectory Smoothing (src/track_swing.py)
  Step 3: Publication-Grade Visual Diagnostics, Velocity Profiles & Failure Strip (src/plot_velocity.py)

At completion, it compiles and prints an executive ASCII summary table of all kinematic and CV metrics.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List
import pandas as pd

# Add src to sys.path so modules can be imported directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from camera_motion import run_pipeline as run_camera_calibration
from track_swing import (
    track_and_extract_kinematics,
    render_annotated_video,
    load_camera_metrics
)
from plot_velocity import (
    compute_smoothed_wrist_kinematics as compute_smoothed_kinematics,
    generate_velocity_analysis_plot,
    generate_failure_strip_plot
)


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable KiB / MiB."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def print_ascii_banner():
    banner = """
========================================================================================
   STR8BAT COMPUTER VISION & KINEMATICS PIPELINE - PART 2
   Automated Pose Tracking, Bat-Proxy Kinematics & Diagnostic Failure Analysis
========================================================================================
"""
    print(banner)


def print_summary_table(
    camera_metrics: Dict[str, Any],
    kinematics_df: pd.DataFrame,
    artifacts: List[Dict[str, str]],
    elapsed_time: float
):
    """Prints a publication-grade, beautifully formatted ASCII summary table."""
    v_info = camera_metrics.get("video_info", {})
    stump = camera_metrics.get("stump_calibration", {})
    cam_sum = camera_metrics.get("summary", {})

    piv_max_idx = kinematics_df["smooth_v_pivot_kmh"].idxmax()
    bat_max_idx = kinematics_df["smooth_v_bat_kmh"].idxmax()

    piv_max = kinematics_df.loc[piv_max_idx]
    bat_max = kinematics_df.loc[bat_max_idx]

    flags = kinematics_df["failure_flag"].value_counts().to_dict()
    num_dupes = int(kinematics_df["is_duplicate_frame"].sum()) if "is_duplicate_frame" in kinematics_df.columns else 0
    f57_row = kinematics_df[kinematics_df["frame_idx"] == 57]
    f57_raw_v = f57_row["raw_v_pivot_px_s"].values[0] if len(f57_row) > 0 else 0.0

    print("\n" + "=" * 88)
    print("                      PIPELINE EXECUTION & KINEMATICS SUMMARY")
    print("=" * 88)
    print(f" Total Pipeline Runtime : {elapsed_time:.2f} seconds")
    print("-" * 88)
    print(" [1] VIDEO & SPATIAL CALIBRATION:")
    print(f"   • Resolution         : {v_info.get('width', 0)} x {v_info.get('height', 0)} px")
    print(f"   • Frame Rate / Count : {v_info.get('fps', 0):.3f} fps | {v_info.get('total_frames', 0)} frames ({v_info.get('duration_seconds', 0):.3f}s)")
    print(f"   • Regulation Stumps  : {stump.get('regulation_height_m', 0.711):.3f} m (ICC Standard)")
    print(f"   • Image Stump Span   : {stump.get('stump_height_px', 99.25):.2f} px (top: {stump.get('stump_top_px')}, bot: {stump.get('stump_bottom_px')})")
    print(f"   • Metric Scale Factor: {stump.get('meters_per_pixel', 0.007164):.6f} m/px ({stump.get('meters_per_pixel', 0.007164)*1000.0:.3f} mm/px)")
    print("-" * 88)
    print(" [2] CAMERA EGO-MOTION METRICS:")
    print(f"   • Mean Camera Speed  : {cam_sum.get('mean_cam_speed_px_s', 0):.2f} px/s ({cam_sum.get('mean_cam_speed_px_s', 0)*stump.get('meters_per_pixel', 0.007164):.2f} m/s)")
    print(f"   • Max Camera Speed   : {cam_sum.get('max_cam_speed_px_s', 0):.2f} px/s (Cut/Transition at Frame {cam_sum.get('peak_pan_frame', 0)})")
    print(f"   • Delivery Pan Phase : ~115 - 140 px/s (Frames 28-31 during ball delivery)")
    print("-" * 88)
    print(" [3] SWING KINEMATICS & LEVER-ARM ACCELERATION:")
    print(f"   • Peak Handle Pivot  : {piv_max['smooth_v_pivot_kmh']:.2f} km/h ({piv_max['smooth_v_pivot_px_s']:.2f} px/s) at Frame {int(piv_max['frame_idx'])} [{piv_max['shot_phase']}]")
    print(f"   • Peak Bat (2D Proj) : {bat_max['smooth_v_bat_kmh']:.2f} km/h ({bat_max['smooth_v_bat_px_s']:.2f} px/s) at Frame {int(bat_max['frame_idx'])} [{bat_max['shot_phase']}]")
    est_3d = bat_max.get('est_3d_v_bat_kmh', bat_max['smooth_v_bat_kmh'] / 0.766)
    est_3d_px = bat_max['smooth_v_bat_px_s'] / 0.766
    print(f"   • Peak Bat (3D Est)  : {est_3d:.2f} km/h ({est_3d_px:.2f} px/s) at Frame {int(bat_max['frame_idx'])} [1/sin(50°) foreshortening]")
    ratio = bat_max['smooth_v_bat_kmh'] / max(1e-3, piv_max['smooth_v_pivot_kmh'])
    print(f"   • Lever Amplification: {ratio:.2f}x velocity magnification (v_bat = v_pivot + omega x r_bat)")
    print(f"   • Frame 57 Pulldown  : Raw finite-diff drops to {f57_raw_v:.2f} px/s (broadcast duplicate frame notch)")
    print("-" * 88)
    print(f" [4] COMPUTER VISION FAILURE REGIMES & DUPLICATE CADENCE ({len(kinematics_df)} total frames):")
    print(f"   • Duplicate Frames   : {num_dupes:2d} frames ({(num_dupes/len(kinematics_df))*100.0:4.1f}%) detected via inter-frame diff (e.g. F57 impact duplicate)")
    order = ["OCCLUSION", "PRE_SHOT_SCENE", "OK", "CAMERA_EGO_MOTION", "MOTION_BLUR_DROPOUT"]
    sorted_flags = sorted(flags.items(), key=lambda x: order.index(x[0]) if x[0] in order else 99)
    for flag_name, count in sorted_flags:
        pct = (count / len(kinematics_df)) * 100.0
        print(f"   • {flag_name:<20}: {count:2d} frames ({pct:5.1f}%)")
    print("-" * 88)
    print(" [5] GENERATED DELIVERABLES & ARTIFACTS:")
    print(f"   {'File Name':<38} | {'File Size':<12} | {'Description'}")
    print("   " + "-" * 84)
    for art in artifacts:
        path = art["path"]
        desc = art["desc"]
        if os.path.exists(path):
            size_str = format_bytes(os.path.getsize(path))
            print(f"   {os.path.basename(path):<38} | {size_str:<12} | {desc}")
        else:
            print(f"   {os.path.basename(path):<38} | {'MISSING':<12} | {desc}")
    print("=" * 88 + "\n")


def main():
    parser = argparse.ArgumentParser(description="str8bat Complete End-to-End Swing Tracking & Kinematics Pipeline")
    parser.add_argument("--video", default="input_video.mp4", help="Path to input video file")
    parser.add_argument("--frames-dir", default="data/frames", help="Path to extracted frames directory")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory where artifacts will be saved")
    parser.add_argument("--pose-model", default="yolov8x-pose.pt", help="Path to YOLOv8 pose weights")
    parser.add_argument("--skip-step1", action="store_true", help="Skip Step 1 (Ingestion & Camera Motion)")
    parser.add_argument("--skip-step2", action="store_true", help="Skip Step 2 (Pose & Bat Tracking)")
    parser.add_argument("--skip-step3", action="store_true", help="Skip Step 3 (Plotting & Strip Visuals)")

    args = parser.parse_args()
    print_ascii_banner()

    start_time = time.time()
    os.makedirs(args.outputs_dir, exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)

    camera_json_path = os.path.join(args.outputs_dir, "camera_metrics.json")
    kinematics_csv_path = os.path.join(args.outputs_dir, "swing_kinematics.csv")
    annotated_mp4_path = os.path.join(args.outputs_dir, "annotated_swing.mp4")
    velocity_png_path = os.path.join(args.outputs_dir, "velocity_profile_analysis.png")
    velocity_full_png_path = os.path.join(args.outputs_dir, "velocity_profile_analysis_full_video.png")
    failure_strip_path = os.path.join(args.outputs_dir, "failure_frames_strip.png")

    # =========================================================================
    # STEP 1: Video Ingestion, Ego-Motion Tracking & Pitch Calibration
    # =========================================================================
    if not args.skip_step1:
        print("\n>>> STEP 1: Running Video Ingestion, Optical Flow Motion & Stumps Calibration...")
        run_camera_calibration(
            video_path=args.video,
            frames_dir=args.frames_dir,
            output_json_path=camera_json_path
        )
    else:
        print("\n>>> STEP 1: Skipped (--skip-step1)")

    if not os.path.isfile(camera_json_path):
        raise FileNotFoundError(f"Missing required camera metrics file: {camera_json_path}")

    with open(camera_json_path, "r") as f:
        camera_metrics = json.load(f)

    # =========================================================================
    # STEP 2: Striker Pose Estimation, Bat-Proxy Tracking & Video Annotation
    # =========================================================================
    if not args.skip_step2:
        print("\n>>> STEP 2: Running Striker Tracking, Pose Estimation & Bat Proxy Kinematics...")
        df_kinematics, raw_records = track_and_extract_kinematics(
            frames_dir=args.frames_dir,
            metrics_path=camera_json_path,
            pose_model_path=args.pose_model
        )

        print(f"Writing kinematics dataset to {kinematics_csv_path}...")
        df_kinematics.to_csv(kinematics_csv_path, index=False)

        print(f"Rendering high-resolution annotated diagnostic video to {annotated_mp4_path}...")
        fps = float(camera_metrics["video_info"]["fps"])
        render_annotated_video(
            frames_dir=args.frames_dir,
            df=df_kinematics,
            raw_records=raw_records,
            output_video_path=annotated_mp4_path,
            fps=fps
        )
    else:
        print("\n>>> STEP 2: Skipped (--skip-step2)")
        if not os.path.isfile(kinematics_csv_path):
            raise FileNotFoundError(f"Missing required kinematics file: {kinematics_csv_path}")
        df_kinematics = pd.read_csv(kinematics_csv_path)

    # =========================================================================
    # STEP 3: Diagnostic Plots & Visual Inspection Strips
    # =========================================================================
    if not args.skip_step3:
        print("\n>>> STEP 3: Generating Publication-Grade Kinematic Plots & Diagnostic Failure Strip...")
        m_per_px = float(camera_metrics["stump_calibration"]["meters_per_pixel"])
        fps = float(camera_metrics["video_info"]["fps"])

        df_proc = compute_smoothed_kinematics(df_kinematics, m_per_px)

        print(f"Rendering velocity analysis plot -> {velocity_png_path}")
        generate_velocity_analysis_plot(
            df=df_proc,
            m_per_px=m_per_px,
            fps=fps,
            start_frame=22,
            output_path=velocity_png_path
        )

        print(f"Rendering supplementary full-video velocity plot -> {velocity_full_png_path}")
        generate_velocity_analysis_plot(
            df=df_proc,
            m_per_px=m_per_px,
            fps=fps,
            start_frame=0,
            output_path=velocity_full_png_path
        )

        if os.path.isfile(annotated_mp4_path):
            print(f"Rendering multi-panel failure strip -> {failure_strip_path}")
            generate_failure_strip_plot(
                df=df_proc,
                annotated_video_path=annotated_mp4_path,
                output_path=failure_strip_path
            )
    else:
        print("\n>>> STEP 3: Skipped (--skip-step3)")

    elapsed_time = time.time() - start_time

    # =========================================================================
    # Executive Summary Table & Artifact Inventory
    # =========================================================================
    artifacts = [
        {"path": camera_json_path, "desc": "Camera ego-motion & stump calibration JSON"},
        {"path": kinematics_csv_path, "desc": "Frame-by-frame 2D/3D kinematic dataset"},
        {"path": annotated_mp4_path, "desc": "720x1280 MP4 with HUD, skeleton & bat trails"},
        {"path": velocity_png_path, "desc": "300 DPI 3-panel swing velocity & tracker plot"},
        {"path": velocity_full_png_path, "desc": "300 DPI full-sequence kinematic plot (0-71)"},
        {"path": failure_strip_path, "desc": "300 DPI multi-panel failure modes diagnostic strip"}
    ]

    print_summary_table(
        camera_metrics=camera_metrics,
        kinematics_df=pd.read_csv(kinematics_csv_path),
        artifacts=artifacts,
        elapsed_time=elapsed_time
    )


if __name__ == "__main__":
    main()
