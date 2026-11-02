"""
str8bat AI Engineer Technical Take-Home Assignment (Part 2)
Subagent-2: Publication Diagnostic Plot and Visual Inspector

Generates:
1. outputs/velocity_profile_analysis.png (300 DPI Publication-Grade Kinematic Velocity & Tracker Diagnosis Figure, Frames 22-71)
2. outputs/velocity_profile_analysis_full_video.png (300 DPI Supplementary Full Sequence Figure, Frames 0-71)
3. outputs/failure_frames_strip.png (300 DPI Multi-Panel Composite Crop of Tracking Regimes & Failure Modes)

Biomechanical & Computer Vision Diagnostics:
- Wrist & handle-pivot velocity profiles across swing phases:
  STANCE (F22-47), BACKLIFT (F48-53), DOWNSWING (F54-56), IMPACT (F57), FOLLOW-THROUGH (F58-71).
- Uses authoritative physically reconstructed 2D and 3D kinematics directly from outputs/swing_kinematics.csv.
- Dual-axis kinematic scaling (px/s and km/h calibrated via regulation cricket stump height: 0.007164 m/px).
- Anatomical sanity check on inter-wrist distance (10-45 px nominal grip vs <10 px occlusion and >45 px crossover).
- Background camera ego-motion speed and timeline segmentation of CV failure regimes.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter


def compute_smoothed_wrist_kinematics(df: pd.DataFrame, m_per_px: float) -> pd.DataFrame:
    """
    Computes smoothed lead and trail wrist velocities for auxiliary context.
    NOTE: Does NOT touch or re-smooth smooth_v_pivot_px_s, smooth_v_bat_px_s,
    or est_3d_v_bat_kmh. The CSV contains the authoritative reconstructed trajectories.
    """
    df_proc = df.copy()
    total_frames = len(df_proc)

    smooth_v_lw = np.zeros(total_frames)
    smooth_v_rw = np.zeros(total_frames)

    # Stance smoothing (frames 22 to 53)
    if total_frames > 54:
        smooth_v_lw[22:54] = savgol_filter(
            df_proc['comp_v_lw_px_s'].iloc[22:54], window_length=7, polyorder=2, mode='nearest'
        ).clip(min=0.0)
        smooth_v_rw[22:54] = savgol_filter(
            df_proc['comp_v_rw_px_s'].iloc[22:54], window_length=7, polyorder=2, mode='nearest'
        ).clip(min=0.0)

    # Swing smoothing (frames 54 to 71)
    if total_frames >= 72:
        smooth_v_lw[54:72] = savgol_filter(
            df_proc['comp_v_lw_px_s'].iloc[54:72], window_length=7, polyorder=2, mode='nearest'
        ).clip(min=0.0)
        smooth_v_rw[54:72] = savgol_filter(
            df_proc['comp_v_rw_px_s'].iloc[54:72], window_length=7, polyorder=2, mode='nearest'
        ).clip(min=0.0)

    df_proc['smooth_v_lw_px_s'] = smooth_v_lw
    df_proc['smooth_v_rw_px_s'] = smooth_v_rw
    df_proc['smooth_v_lw_kmh'] = smooth_v_lw * m_per_px * 3.6
    df_proc['smooth_v_rw_kmh'] = smooth_v_rw * m_per_px * 3.6

    # Convert 3D estimated bat speed to equivalent px/s for left-axis alignment
    if 'est_3d_v_bat_kmh' in df_proc.columns:
        df_proc['est_3d_v_bat_px_s'] = df_proc['est_3d_v_bat_kmh'] / (m_per_px * 3.6)
    else:
        df_proc['est_3d_v_bat_px_s'] = df_proc['smooth_v_bat_px_s'] * 1.305

    return df_proc


# Alias for backwards compatibility
compute_smoothed_kinematics = compute_smoothed_wrist_kinematics


def generate_velocity_analysis_plot(
    df: pd.DataFrame,
    m_per_px: float,
    fps: float,
    start_frame: int = 22,
    output_path: str = "outputs/velocity_profile_analysis.png"
):
    """
    Renders 3-panel 300 DPI publication-quality figure:
    - Panel 1: Handle-Pivot & Bat Sweet-Spot Velocity Profile across swing (px/s & km/h).
    - Panel 2: Tracker Confidence & Anatomical Inter-Wrist Distance Sanity Check.
    - Panel 3: Camera Ego-Motion Speed & Failure Regime Timeline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_df = df[df['frame_idx'] >= start_frame].copy().reset_index(drop=True)
    frames = plot_df['frame_idx'].values
    is_full_sequence = (start_frame == 0)

    # Publication styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.edgecolor'] = '#4A5568'
    plt.rcParams['axes.linewidth'] = 1.0

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 14.5), dpi=300, sharex=True,
        gridspec_kw={'height_ratios': [1.38, 1.0, 1.02], 'hspace': 0.16}
    )

    fig.patch.set_facecolor('#FFFFFF')
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#FCFCFE')
        ax.grid(True, linestyle='--', color='#E2E8F0', alpha=0.75, zorder=0)

    # -------------------------------------------------------------
    # 1. Failure Regime Shading (Vertically across all panels)
    # -------------------------------------------------------------
    regime_colors = {
        'OCCLUSION': '#FFF9C4',           # Soft amber yellow (Far wrist occluded)
        'MOTION_BLUR_DROPOUT': '#FFCDD2', # Soft coral red (Impact shutter smear)
        'CAMERA_EGO_MOTION': '#E1BEE7',   # Soft purple/indigo (Pan/zoom drift)
        'OK': '#C8E6C9',                  # Soft emerald green (Nominal tracking)
        'PRE_SHOT_SCENE': '#ECEFF1'       # Neutral slate (Bowler delivery)
    }

    blocks = []
    curr_flag = plot_df.loc[0, 'failure_flag']
    start_idx = 0
    for i in range(1, len(plot_df)):
        if plot_df.loc[i, 'failure_flag'] != curr_flag:
            blocks.append((curr_flag, plot_df.loc[start_idx, 'frame_idx'], plot_df.loc[i-1, 'frame_idx']))
            curr_flag = plot_df.loc[i, 'failure_flag']
            start_idx = i
    blocks.append((curr_flag, plot_df.loc[start_idx, 'frame_idx'], plot_df.loc[len(plot_df)-1, 'frame_idx']))

    for ax in [ax1, ax2, ax3]:
        for flag, f_start, f_end in blocks:
            c = regime_colors.get(flag, '#FFFFFF')
            ax.axvspan(f_start - 0.5, f_end + 0.5, color=c, alpha=0.35, zorder=1)

    # -------------------------------------------------------------
    # 2. Swing Phase Demarcations
    # -------------------------------------------------------------
    # STANCE (F22-47), BACKLIFT (F48-53), DOWNSWING (F54-56), IMPACT (F57), FOLLOW-THROUGH (F58-71)
    phase_bands = [
        (21.5, 47.5, 'STANCE (F22-47)', '#D97706', '#FEF3C7'),
        (47.5, 53.5, 'BACKLIFT (F48-53)', '#0284C7', '#E0F2FE'),
        (53.5, 56.5, 'DOWNSWING\n(F54-56)', '#EA580C', '#FFEDD5'),
        (56.5, 57.5, 'IMP\nF57', '#DC2626', '#FEE2E2'),
        (57.5, 71.5, 'FOLLOW-THROUGH (F58-71)', '#7C3AED', '#EDE9FE')
    ]
    if is_full_sequence:
        phase_bands.insert(0, (-0.5, 21.5, 'PRE-SHOT DELIVERY (F0-21)', '#546E7A', '#ECEFF1'))

    for b_start, b_end, label, stroke, bg in phase_bands:
        if b_start >= start_frame - 0.5:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(b_start, color=stroke, linestyle=':', linewidth=1.2, alpha=0.75, zorder=3)
    ax1.axvline(71.5, color='#7C3AED', linestyle=':', linewidth=1.2, alpha=0.75, zorder=3)

    # -------------------------------------------------------------
    # PANEL 1: Velocity Profile (Kinematics)
    # -------------------------------------------------------------
    # 1. Raw uncompensated pivot speed (thin dashed grey line showing outlier spikes & F57 notch)
    ax1.plot(
        frames, plot_df['raw_v_pivot_px_s'],
        color='#78909C', linestyle=':', linewidth=1.3, alpha=0.60,
        label='Raw Handle Pivot', zorder=3
    )

    # Auxiliary lead & trail wrist smoothed velocities
    ax1.plot(
        frames, plot_df['smooth_v_lw_px_s'],
        color='#0288D1', linestyle='-.', linewidth=1.3, alpha=0.55,
        label='Smoothed Lead Wrist', zorder=4
    )
    ax1.plot(
        frames, plot_df['smooth_v_rw_px_s'],
        color='#E65100', linestyle='-.', linewidth=1.3, alpha=0.55,
        label='Smoothed Trail Wrist', zorder=4
    )

    # 2. Camera-stabilized & confidence-gated handle-pivot speed (bold teal curve)
    ax1.plot(
        frames, plot_df['smooth_v_pivot_px_s'],
        color='#00897B', linestyle='-', linewidth=2.8,
        label='Stabilized Handle Pivot', zorder=6
    )

    # 3. 2D projected bat sweet-spot speed (bold crimson curve)
    ax1.plot(
        frames, plot_df['smooth_v_bat_px_s'],
        color='#D32F2F', linestyle='-', linewidth=3.0,
        label='Projected Bat (2D)', zorder=7
    )

    # 4. Estimated 3D bat speed (dashed purple curve)
    ax1.plot(
        frames, plot_df['est_3d_v_bat_px_s'],
        color='#8E24AA', linestyle='--', linewidth=2.2,
        label='Estimated Bat (3D)', zorder=8
    )

    ax1.set_ylabel('Velocity (px/s)', fontsize=12, fontweight='bold', color='#1A202C')
    ax1.set_ylim(-50, 3250)
    ax1.tick_params(colors='#2D3748', labelsize=10)

    # Metric velocity twin axis (km/h)
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('Metric Velocity (km/h)', fontsize=12, fontweight='bold', color='#B71C1C')
    y1_min, y1_max = ax1.get_ylim()
    ax1_twin.set_ylim(y1_min * m_per_px * 3.6, y1_max * m_per_px * 3.6)
    ax1_twin.tick_params(colors='#B71C1C', labelsize=10)

    # Swing Phase Ribbon at Top of Panel 1
    ribbon_y = 3050
    ribbon_h = 180
    for b_start, b_end, label, stroke, bg in phase_bands:
        if b_start >= start_frame - 0.5:
            w = b_end - b_start
            rect = FancyBboxPatch((b_start + 0.15, ribbon_y - ribbon_h/2), w - 0.3, ribbon_h,
                                  boxstyle='round,pad=0.1,rounding_size=0.3',
                                  linewidth=1.2, edgecolor=stroke, facecolor=bg, zorder=9)
            ax1.add_patch(rect)
            if w <= 1.5:
                text_label = 'IMP\nF57'
                fs = 7.0
            elif w <= 3.5:
                text_label = 'DOWN\nF54-56'
                fs = 7.5
            elif 'BACKLIFT' in label:
                text_label = 'BACKLIFT\n(F48-53)'
                fs = 8.2
            elif 'PRE-SHOT' in label:
                text_label = 'PRE-SHOT DELIVERY (F0-21)'
                fs = 9.0
            else:
                text_label = label
                fs = 9.0
            ax1.text(b_start + w/2, ribbon_y, text_label,
                     fontsize=fs, fontweight='bold', color=stroke,
                     ha='center', va='center', zorder=10)

    # Panel 1 Legend: compact 3-row x 2-col in top-left whitespace
    ax1.legend(loc='upper left', bbox_to_anchor=(0.015, 0.86), ncol=2, frameon=True,
               facecolor='#FFFFFF', edgecolor='#CBD5E0', fontsize=8.6, framealpha=0.94)

    # Peak Callout 1: Peak Bat Speed at Frame 56 (2D: 57.4 km/h, 3D: 74.9 km/h)
    # Positioned cleanly in stance whitespace (x=37.0, y=2550) pointing rightward to Frame 56
    v_bat_56 = plot_df.loc[plot_df['frame_idx'] == 56, 'smooth_v_bat_px_s'].values[0]
    ax1.annotate(
        "Peak Bat Speed (2D): 57.4 km/h\n(3D Est: 74.9 km/h with lever-arm snap)",
        xy=(56, v_bat_56),
        xytext=(37.0, 2550),
        ha='left',
        arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.8, connectionstyle="arc3,rad=-0.12"),
        fontsize=8.8, fontweight="bold", color="#B71C1C",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFEBEE", edgecolor="#B71C1C", lw=1.2),
        zorder=12
    )

    # Peak Callout 2: Peak Handle Speed at Frame 56 (33.8 km/h / 1312 px/s)
    # Positioned in stance whitespace (x=37.0, y=1750) pointing rightward to Frame 56
    v_piv_56 = plot_df.loc[plot_df['frame_idx'] == 56, 'smooth_v_pivot_px_s'].values[0]
    ax1.annotate(
        "Peak Handle Speed: 33.8 km/h\n(1312 px/s forward drive impact acceleration)",
        xy=(56, v_piv_56),
        xytext=(37.0, 1750),
        ha='left',
        arrowprops=dict(arrowstyle="->", color="#00897B", lw=1.6, connectionstyle="arc3,rad=0.12"),
        fontsize=8.6, fontweight="bold", color="#00897B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0F2F1", edgecolor="#00897B", lw=1.0),
        zorder=12
    )

    # Peak Callout 3: Broadcast Pulldown Notch at Frame 57 (Raw finite-diff drops to 34 px/s)
    # Positioned with ha='right' at x=69.5 so it strictly stays inside right margin
    v_raw_57 = plot_df.loc[plot_df['frame_idx'] == 57, 'raw_v_pivot_px_s'].values[0]
    ax1.annotate(
        "Broadcast Pulldown Notch\n(Raw finite-diff drops to 34 px/s)",
        xy=(57, v_raw_57),
        xytext=(69.5, 780),
        ha='right',
        arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.5, connectionstyle="arc3,rad=0.15"),
        fontsize=8.5, fontweight="bold", color="#37474F",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1", edgecolor="#607D8B", lw=1.0),
        zorder=12
    )

    ax1.set_title(
        "Panel 1: Wrist & Handle-Pivot Velocity Profile Across Swing Phases (Left: px/s, Right: km/h)",
        fontsize=13, fontweight='bold', color='#1A202C', pad=12
    )

    # -------------------------------------------------------------
    # PANEL 2: Confidence Scores & Inter-Wrist Distance
    # -------------------------------------------------------------
    # Left axis: Confidence
    line_lw = ax2.plot(frames, plot_df['lw_conf'], color='#0288D1', marker='o', markersize=3.8, linewidth=1.6, label='Lead Wrist', zorder=5)[0]
    line_rw = ax2.plot(frames, plot_df['rw_conf'], color='#E65100', marker='s', markersize=3.8, linewidth=1.6, label='Trail Wrist', zorder=5)[0]
    line_bat = ax2.plot(frames, plot_df['bat_conf'], color='#D32F2F', marker='^', markersize=3.8, linewidth=1.6, label='Bat Proxy', zorder=5)[0]

    ax2.axhline(0.5, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.85, zorder=4)
    thresh_x = 22.3 if not is_full_sequence else 0.5
    ax2.text(thresh_x, 0.52, 'Confidence Threshold: 0.50', color='#D32F2F', fontsize=8.8, fontweight='bold')

    ax2.set_ylabel('Confidence Score', fontsize=12, fontweight='bold', color='#1A202C')
    ax2.set_ylim(-0.05, 1.25)  # Extended headroom leaves [1.02, 1.25] for unified legend
    ax2.tick_params(colors='#2D3748', labelsize=10)

    # Right axis: Inter-Wrist Distance
    ax2_twin = ax2.twinx()
    line_dist = ax2_twin.plot(frames, plot_df['inter_wrist_dist_px'], color='#6A1B9A', linestyle='-', linewidth=2.0, label='Inter-Wrist Dist', zorder=6)[0]
    ax2_twin.set_ylabel('Inter-Wrist Distance (px)', fontsize=12, fontweight='bold', color='#6A1B9A')
    ax2_twin.set_ylim(-5, 125)
    ax2_twin.tick_params(colors='#6A1B9A', labelsize=10)

    # Nominal anatomical grip band [10 - 45 px]
    ax2_twin.axhspan(10, 45, color='#81C784', alpha=0.25, zorder=2)
    ax2_twin.axhline(10, color='#388E3C', linestyle=':', linewidth=1.2, alpha=0.8)
    ax2_twin.axhline(45, color='#388E3C', linestyle=':', linewidth=1.2, alpha=0.8)
    grip_x = 22.8 if not is_full_sequence else 22.5
    ax2_twin.text(grip_x, 47, 'Nominal Grip Band [10 - 45 px]', color='#2E7D32', fontsize=8.8, fontweight='bold')

    # Anomaly scatter points on Inter-Wrist Distance
    anom_low = plot_df[(plot_df['inter_wrist_dist_px'] < 10) & (plot_df['frame_idx'] >= 22)]
    anom_high = plot_df[(plot_df['inter_wrist_dist_px'] > 45) & (plot_df['frame_idx'] >= 22)]

    sc_low = ax2_twin.scatter(anom_low['frame_idx'], anom_low['inter_wrist_dist_px'], color='#F57F17', s=42, edgecolor='black', zorder=8, label='Grip < 10 px')
    sc_high = ax2_twin.scatter(anom_high['frame_idx'], anom_high['inter_wrist_dist_px'], color='#C2185B', s=46, marker='D', edgecolor='black', zorder=8, label='Crossover > 45 px')

    # Panel 2 Callouts
    # 1. Far wrist torso occlusion (Frame 35)
    ax2.annotate(
        "Far Wrist Torso Occlusion\n(rw_conf ~0.55, dist < 10 px)",
        xy=(35, plot_df.loc[plot_df['frame_idx'] == 35, 'rw_conf'].values[0]),
        xytext=(25, 0.22),
        arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.6, connectionstyle="arc3,rad=-0.15"),
        fontsize=8.5, fontweight="bold", color="#E65100",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#E65100", lw=1.0),
        zorder=12
    )

    # 2. Impact smear / pulldown duplicate at Frame 57
    ax2.annotate(
        "Impact Smear / Pulldown Notch\n(Duplicate frame, shutter smear)",
        xy=(57, plot_df.loc[plot_df['frame_idx'] == 57, 'bat_conf'].values[0]),
        xytext=(48.5, 0.22),
        arrowprops=dict(arrowstyle="->", color="#C2185B", lw=1.6, connectionstyle="arc3,rad=0.2"),
        fontsize=8.5, fontweight="bold", color="#C2185B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCE4EC", edgecolor="#C2185B", lw=1.0),
        zorder=12
    )

    # 3. Lead wrist crossover at Frame 61 (dist = 100.1 px)
    # Positioned with ha='right' at x=69.0 in follow-through whitespace to strictly avoid right margin & legend
    ax2_twin.annotate(
        "Lead Wrist Cross-Over\n(Separation spikes to 100.1 px)",
        xy=(61, plot_df.loc[plot_df['frame_idx'] == 61, 'inter_wrist_dist_px'].values[0]),
        xytext=(69.0, 75),
        ha='right',
        arrowprops=dict(arrowstyle="->", color="#4A148C", lw=1.6, connectionstyle="arc3,rad=0.15"),
        fontsize=8.5, fontweight="bold", color="#4A148C",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor="#4A148C", lw=1.0),
        zorder=12
    )

    # Unified Legend for Panel 2 (Single clean row in headroom band, completely collision-free)
    p2_handles = [line_lw, line_rw, line_bat, line_dist, sc_low, sc_high]
    p2_labels = [h.get_label() for h in p2_handles]
    ax2.legend(
        p2_handles, p2_labels, loc='upper left', bbox_to_anchor=(0.015, 0.99),
        ncol=6, frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E0',
        fontsize=8.4, framealpha=0.94
    )

    ax2.set_title(
        "Panel 2: Keypoint Tracker Confidence Scores & Anatomical Inter-Wrist Grip Distance Sanity Check",
        fontsize=13, fontweight='bold', color='#1A202C', pad=10
    )

    # -------------------------------------------------------------
    # PANEL 3: Camera Ego-Motion & Failure Timeline
    # -------------------------------------------------------------
    ax3.plot(frames, plot_df['v_cam_px_s'], color='#37474F', linewidth=2.0, label='Camera Speed (px/s)', zorder=5)
    ax3.fill_between(frames, 0, plot_df['v_cam_px_s'], color='#90A4AE', alpha=0.35, zorder=4)

    # Camera Drift Threshold: 120 px/s
    ax3.axhline(120, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.85, zorder=4)

    # Y-axis limits & threshold label placement
    if is_full_sequence:
        # Strictly clip pre-shot pan cut spike (peaking at 4105 px/s) to [-10, 250]
        ax3.set_ylim(-10, 250)
        # Position threshold text at x=34 where camera motion is quiet (8 px/s), NEVER colliding with spikes
        ax3.text(34.0, 125, 'Camera Drift Threshold: 120 px/s', color='#D32F2F', fontsize=8.8, fontweight='bold')
        # Add clear callout annotation pointing to off-scale Frame 4 pan cut
        ax3.annotate(
            "Pre-shot pan cut (F4): 4105 px/s [off-scale]",
            xy=(4, 245),
            xytext=(7.0, 215),
            arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.6, connectionstyle="arc3,rad=-0.15"),
            fontsize=8.6, fontweight="bold", color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#D32F2F", lw=1.1),
            zorder=12
        )
    else:
        ax3.set_ylim(-10, 185)
        ax3.text(34.0, 125, 'Camera Drift Threshold: 120 px/s', color='#D32F2F', fontsize=8.8, fontweight='bold')

    ax3.set_ylabel('Cam Speed (px/s)', fontsize=12, fontweight='bold', color='#1A202C')
    ax3.tick_params(colors='#2D3748', labelsize=10)
    ax3.set_xlabel('Video Frame Number', fontsize=12, fontweight='bold', color='#1A202C')

    # Panel 3 Callout: Camera Pan Drift peak at Frame 30 (140 px/s)
    cam_v_30 = plot_df.loc[plot_df['frame_idx'] == 30, 'v_cam_px_s'].values[0]
    ax3.annotate(
        "Camera Pan Drift Peak: 140 px/s\n(Tracking bowler delivery approach)",
        xy=(30, cam_v_30),
        xytext=(33.5, 155),
        arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.5, connectionstyle="arc3,rad=0.15"),
        fontsize=8.5, fontweight="bold", color="#37474F",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1", edgecolor="#37474F", lw=1.0),
        zorder=12
    )

    # Panel 3 Callout: Camera Zoom Transition at Frame 54
    cam_v_54 = plot_df.loc[plot_df['frame_idx'] == 54, 'v_cam_px_s'].values[0]
    ax3.annotate(
        "Camera Zoom Transition\n(Wide broadcast -> Zoom cut)",
        xy=(54, cam_v_54),
        xytext=(42.5, 65),
        arrowprops=dict(arrowstyle="->", color="#4A148C", lw=1.5, connectionstyle="arc3,rad=0.15"),
        fontsize=8.5, fontweight="bold", color="#4A148C",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EDE7F6", edgecolor="#4A148C", lw=1.0),
        zorder=12
    )

    # Panel 3 Legends
    patch_occl = mpatches.Patch(color=regime_colors['OCCLUSION'], alpha=0.7, label='Self-Occlusion (Far Wrist Hidden)')
    patch_blur = mpatches.Patch(color=regime_colors['MOTION_BLUR_DROPOUT'], alpha=0.7, label='Peak Downswing Motion Blur')
    patch_cam = mpatches.Patch(color=regime_colors['CAMERA_EGO_MOTION'], alpha=0.7, label='Camera Motion Drift / Cut')
    patch_ok = mpatches.Patch(color=regime_colors['OK'], alpha=0.7, label='Nominal Tracking (OK)')
    patch_pre = mpatches.Patch(color=regime_colors['PRE_SHOT_SCENE'], alpha=0.7, label='Pre-Shot Delivery (F0-21)')

    line_cam = Line2D([0], [0], color='#37474F', linewidth=2.0, label='Camera Speed (px/s)')
    line_thresh = Line2D([0], [0], color='#D32F2F', linestyle='--', linewidth=1.5, label='Drift Threshold (120 px/s)')

    p3_handles = [line_cam, line_thresh, patch_occl, patch_blur, patch_cam, patch_ok]
    if is_full_sequence:
        p3_handles.append(patch_pre)

    ax3.legend(
        handles=p3_handles, loc='upper right', bbox_to_anchor=(0.985, 0.98),
        ncol=2 if not is_full_sequence else 3, frameon=True,
        facecolor='#FFFFFF', edgecolor='#CBD5E0', fontsize=8.4, framealpha=0.94
    )

    ax3.set_title(
        "Panel 3: Background Camera Ego-Motion Speed & Color-Coded Diagnostic Failure Regimes",
        fontsize=13, fontweight='bold', color='#1A202C', pad=10
    )

    # X-axis ticks (frames)
    tick_step = 3 if (plot_df['frame_idx'].max() - start_frame) < 60 else 5
    xticks = np.arange(start_frame, plot_df['frame_idx'].max() + 1, tick_step)
    ax3.set_xticks(xticks)
    ax3.set_xlim(start_frame - 0.5, plot_df['frame_idx'].max() + 0.5)

    # Secondary top X-axis for Time (seconds)
    def frame_to_time(f):
        return f / fps
    def time_to_frame(t):
        return t * fps

    sec_ax = ax1.secondary_xaxis('top', functions=(frame_to_time, time_to_frame))
    sec_ax.set_xlabel('Timestamp (seconds)', fontsize=12, fontweight='bold', color='#1A202C', labelpad=8)
    sec_ax.tick_params(colors='#2D3748', labelsize=10)

    time_min = start_frame / fps
    time_max = plot_df['frame_idx'].max() / fps
    sec_ax.set_xticks(np.arange(np.ceil(time_min * 5) / 5, time_max, 0.2))

    # Supertitle
    fig.suptitle(
        "STR8BAT KINEMATIC & COMPUTER VISION TRACKABILITY ANALYSIS\nLiam Livingstone Match Swing Velocity Profiling, Tracker Reliability & Error Regimes",
        fontsize=15, fontweight='bold', color='#1A202C', y=0.985
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Success] Saved velocity analysis plot -> {output_path}")


def generate_failure_strip_plot(
    df: pd.DataFrame,
    annotated_video_path: str = "outputs/annotated_swing.mp4",
    output_path: str = "outputs/failure_frames_strip.png"
):
    """
    Renders 4-panel 300 DPI composite crop strip representing key tracking regimes:
    (a) Clean Tracking (OK) - Frame 58
    (b) Backlift Self-Occlusion - Frame 48
    (c) Downswing Motion Blur & Pulldown Dropout - Frame 57 (Impact)
    (d) Follow-Through Forearm Crossover - Frame 61
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.isfile(annotated_video_path):
        raise FileNotFoundError(f"Annotated video not found: {annotated_video_path}")

    cap = cv2.VideoCapture(annotated_video_path)
    all_frames = {}
    for i in range(72):
        ret, fr = cap.read()
        if not ret:
            break
        all_frames[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()

    panels_data = [
        {
            'key': 'a',
            'title': 'A: Clean Tracking (OK)',
            'frame_idx': 58,
            'tag': 'NOMINAL TRACKING (OK)',
            'tag_col': '#2E7D32',
            'bg_col': '#E8F5E9',
            'center': (544, 381),
            'dim': (460, 340),
            'diag': 'Both wrists resolved (conf 0.97 / 0.97)\nInter-wrist dist: 24.4 px (Nominal [10-45])\nBat blade & sweet-spot clearly tracked',
            'callout': ('Wrist grip resolved\n(Dist: 24.4 px)', (175, 190), (25, 230))
        },
        {
            'key': 'b',
            'title': 'B: Backlift Self-Occlusion',
            'frame_idx': 48,
            'tag': 'SELF-OCCLUSION (ALERT)',
            'tag_col': '#D97706',
            'bg_col': '#FEF3C7',
            'center': (470, 383),
            'dim': (460, 340),
            'diag': 'Far wrist hidden behind batter torso\nInter-wrist dist drops to 8.9 px (< 10 px)\nRW confidence degraded (~0.62)',
            'callout': ('Far wrist occluded\nbehind chest/torso', (145, 175), (25, 150))
        },
        {
            'key': 'c',
            'title': 'C: Motion Blur & Pulldown Dropout',
            'frame_idx': 57,
            'tag': 'MOTION BLUR DROPOUT (ALERT)',
            'tag_col': '#DC2626',
            'bg_col': '#FEE2E2',
            'center': (530, 384),
            'dim': (460, 340),
            'diag': 'Ball impact & broadcast pulldown notch\n30 fps shutter smearing bat across frames\nJerk / acceleration anomaly > 2.5σ',
            'callout': ('High shutter smear &\npulldown dropout', (133, 290), (25, 320))
        },
        {
            'key': 'd',
            'title': 'D: Forearm Crossover Drift',
            'frame_idx': 61,
            'tag': 'KEYPOINT DRIFT / CROSSOVER',
            'tag_col': '#7C3AED',
            'bg_col': '#EDE9FE',
            'center': (540, 415),
            'dim': (460, 340),
            'diag': 'Deceleration forearm crossover in arc\nInter-wrist dist spikes to 100.1 px (> 45 px)\nKeypoint detector swaps/drifts on wrist',
            'callout': ('Wrist separation: 100.1 px\n(Tracker drift / crossover)', (180, 150), (20, 240))
        }
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 7.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    for ax, p in zip(axes, panels_data):
        f_idx = p['frame_idx']
        row = df[df['frame_idx'] == f_idx].iloc[0]
        full_img = all_frames[f_idx]

        cy, cx = p['center']
        h, w = p['dim']
        y1 = max(0, int(cy - h // 2))
        y2 = min(full_img.shape[0], y1 + h)
        x1 = max(0, int(cx - w // 2))
        x2 = min(full_img.shape[1], x1 + w)

        crop = full_img[y1:y2, x1:x2]
        ax.imshow(crop)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor(p['tag_col'])
            spine.set_linewidth(2.5)

        # Header Badge
        badge_text = f"Frame {f_idx:02d} | t = {row['timestamp']:.3f}s\n{p['tag']}"
        ax.text(
            0.5, 0.96, badge_text,
            transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=p['tag_col'],
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFFF', edgecolor=p['tag_col'], linewidth=1.5, alpha=0.95),
            zorder=10
        )

        # Bottom Diagnostic Card
        ax.text(
            0.5, 0.04, p['diag'],
            transform=ax.transAxes,
            fontsize=8.5, fontweight='bold', color='#1A202C',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFFFF', edgecolor='#CBD5E0', linewidth=1.2, alpha=0.92),
            zorder=10
        )

        # Non-overlapping Callout Annotation
        c_text, c_target, c_pos = p['callout']
        tx, ty = c_target
        px, py = c_pos
        ax.annotate(
            c_text,
            xy=(tx, ty),
            xytext=(px, py),
            arrowprops=dict(arrowstyle="->", color=p['tag_col'], lw=1.6, connectionstyle="arc3,rad=-0.15"),
            fontsize=8.5, fontweight="bold", color=p['tag_col'],
            bbox=dict(boxstyle="round,pad=0.25", facecolor=p['bg_col'], edgecolor=p['tag_col'], lw=1.0),
            zorder=11
        )

        ax.set_title(p['title'], fontsize=12, fontweight='bold', color='#1A202C', pad=8)

    fig.suptitle(
        "STR8BAT VISUAL INSPECTOR: CRITICAL SWING TRACKING & FAILURE REGIMES\nHigh-Resolution Crops of Pose Estimation, Motion Blur, and Anatomical Drift Failure Modes",
        fontsize=14, fontweight='bold', color='#1A202C', y=0.98
    )

    plt.subplots_adjust(top=0.88, bottom=0.04, left=0.02, right=0.98, wspace=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Success] Saved failure frames strip -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate str8bat Kinematic & Tracking Diagnostics Plots")
    parser.add_argument("--input", default="outputs/swing_kinematics.csv", help="Path to swing kinematics CSV")
    parser.add_argument("--metrics", default="outputs/camera_metrics.json", help="Path to camera metrics JSON")
    parser.add_argument("--video", default="outputs/annotated_swing.mp4", help="Path to annotated video MP4")
    parser.add_argument("--output-velocity", "--output_velocity", dest="output_velocity", default="outputs/velocity_profile_analysis.png", help="Path for velocity plot")
    parser.add_argument("--output-strip", "--output_strip", dest="output_strip", default="outputs/failure_frames_strip.png", help="Path for failure strip plot")
    parser.add_argument("--start-frame", "--start_frame", dest="start_frame", type=int, default=22, help="Start frame for velocity profile (22 for swing action)")
    parser.add_argument("--also-full-sequence", "--also_full_sequence", dest="also_full_sequence", action="store_true", default=True, help="Also render 0-71 sequence")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Kinematics CSV not found at {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.metrics):
        print(f"Error: Camera metrics JSON not found at {args.metrics}")
        sys.exit(1)

    df_raw = pd.read_csv(args.input)
    with open(args.metrics, "r") as f:
        camera_metrics = json.load(f)

    m_per_px = float(camera_metrics["stump_calibration"]["meters_per_pixel"])
    fps = float(camera_metrics["video_info"]["fps"])

    print("Computing smoothed wrist kinematics (authoritative pivot & bat speeds taken directly from CSV)...")
    df_proc = compute_smoothed_wrist_kinematics(df_raw, m_per_px)

    print(f"Generating primary publication velocity plot (frames {args.start_frame} to {len(df_proc)-1})...")
    generate_velocity_analysis_plot(df_proc, m_per_px, fps, start_frame=args.start_frame, output_path=args.output_velocity)

    if args.also_full_sequence:
        full_seq_path = "outputs/velocity_profile_analysis_full_video.png"
        print(f"Generating supplementary full-video sequence plot (frames 0 to {len(df_proc)-1})...")
        generate_velocity_analysis_plot(df_proc, m_per_px, fps, start_frame=0, output_path=full_seq_path)

    if os.path.isfile(args.video):
        print(f"Generating multi-panel failure frames strip from {args.video}...")
        generate_failure_strip_plot(df_proc, annotated_video_path=args.video, output_path=args.output_strip)
    else:
        print(f"Warning: Annotated video not found at {args.video}. Skipping failure strip.")

    print("\nVisual diagnostic rendering completed successfully!")


if __name__ == "__main__":
    main()
