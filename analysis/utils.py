# Copyright (c) 2024 TACTICS. All Rights Reserved.
#
# 이 소프트웨어의 상업적 사용, 수정 및 배포를 금지합니다.
# 허가 없이 이 코드를 사용, 복제, 수정, 배포할 수 없습니다.
#
# Commercial use, modification, and distribution of this software are prohibited.
# You may not use, copy, modify, or distribute this code without permission.

import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile

# Ensure torch and ultralytics are imported as they are used by analyze_video
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ultralytics import YOLO

# --- PyInstaller: Force MediaPipe to use bundled models ---
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    os.environ['MEDIAPIPE_MODEL_CACHE_DIR'] = os.path.join(sys._MEIPASS, 'mediapipe', 'modules')

# =================================================================================
# Constants
# =================================================================================

# --- MediaPipe Landmark Indices ---
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW = 0, 11, 12, 14
LEFT_WRIST, RIGHT_WRIST, LEFT_INDEX, RIGHT_INDEX = 15, 16, 19, 20
LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = 23, 24, 25, 26, 27, 28
LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 29, 30, 31, 32

# --- Skeleton Rendering ---
LEFT_EDGES = [(11, 23), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31)]
RIGHT_EDGES = [(12, 14), (14, 16), (16, 22), (16, 18), (16, 20), (18, 20),
               (12, 24), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]
CENTER_EDGES = [(11, 12), (23, 24)]

# --- Physics and Analysis ---
SHIN_LENGTH_M = 0.45
VIDEO_FPS = 120  # Default FPS if not detected from video


# =================================================================================
# Custom Exception
# =================================================================================

class PhaseSegmentationError(Exception):
    """Exception raised for errors in the video phase segmentation process."""
    pass


# =================================================================================
# Django-related Utilities
# =================================================================================

def create_thumbnail_from_video(video_file):
    """
    Extracts the first frame from a video file and returns it as a Django ContentFile.
    """
    temp_video_path = os.path.join(settings.MEDIA_ROOT, 'temp_video.mp4')
    with open(temp_video_path, 'wb+') as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    cap = cv2.VideoCapture(temp_video_path)
    success, frame = cap.read()
    cap.release()
    os.remove(temp_video_path)

    if not success:
        return None

    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        return None

    return ContentFile(buffer.tobytes())


# =================================================================================
# Core Analysis Functions
# =================================================================================

def analyze_video(video_path, yolo_model):
    """
    Analyzes a video to find key pitching frames (start, max knee height, foot plant, release).
    This is the primary function for video segmentation and also calculates
    wrist speed and shoulder angular velocity.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or VIDEO_FPS  # Get actual FPS, fallback to constant

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2,
                                  smooth_landmarks=True, min_detection_confidence=0.7,
                                  min_tracking_confidence=0.5)

    frames, landmarks_list = [], []
    knee_y_list, ankle_y_list, ankle_x_list = [], [], []
    foot_x_list, foot_y_list = [], []

    SHIN_SCALE_DIVISOR = 10

    while True:
        ret, frame = cap.read()
        if not ret: break

        frames.append(frame)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            knee_y_list.append(int(lm[LEFT_KNEE].y * height))
            ankle_y_list.append(int(lm[LEFT_ANKLE].y * height))
            ankle_x_list.append(int(lm[LEFT_ANKLE].x * width))
            foot_x_list.append(int(lm[LEFT_FOOT_INDEX].x * width))
            foot_y_list.append(int(lm[LEFT_FOOT_INDEX].y * height))
            landmarks_list.append(lm)
        else:
            knee_y_list.append(None)
            ankle_y_list.append(None)
            ankle_x_list.append(None)
            foot_x_list.append(None)
            foot_y_list.append(None)
            landmarks_list.append(None)
    cap.release()

    total_frames = len(frames)
    valid_knees = [(i, y) for i, y in enumerate(knee_y_list) if y is not None]
    max_knee_frame = min(valid_knees, key=lambda x: x[1])[0] if valid_knees else None

    threshold = None
    if max_knee_frame is not None and landmarks_list[max_knee_frame] is not None:
        lm = landmarks_list[max_knee_frame]
        rk, ra = lm[RIGHT_KNEE], lm[RIGHT_ANKLE]
        x1, y1 = rk.x * width, rk.y * height
        x2, y2 = ra.x * width, ra.y * height
        threshold = math.hypot(x2 - x1, y2 - y1) / SHIN_SCALE_DIVISOR

    start_frame, fixed_frame, release_frame = None, None, None

    if threshold and max_knee_frame is not None:
        count = 0
        for i in range(max_knee_frame - 5):
            if i + 5 < len(ankle_y_list) and ankle_y_list[i] and ankle_y_list[i + 5]:
                if abs(ankle_y_list[i + 5] - ankle_y_list[i]) > threshold:
                    count += 1
                    if count == 5:
                        start_frame = max(0, i - 20)
                        break
                else:
                    count = 0

    if max_knee_frame and threshold:
        epsilon = 0.8 * threshold
        for i in range(max_knee_frame, total_frames - 15):
            valid = True
            for j in range(5):
                f1, f2 = i + j, i + j + 10
                if f2 >= total_frames or None in [
                    ankle_x_list[f1], ankle_x_list[f2], ankle_y_list[f1], ankle_y_list[f2],
                    foot_x_list[f1], foot_x_list[f2], foot_y_list[f1], foot_y_list[f2]
                ]:
                    valid = False
                    break
                if (abs(ankle_x_list[f2] - ankle_x_list[f1]) > epsilon or
                        abs(ankle_y_list[f2] - ankle_y_list[f1]) > epsilon or
                        abs(foot_x_list[f2] - foot_x_list[f1]) > epsilon or
                        abs(foot_y_list[f2] - foot_y_list[f1]) > epsilon):
                    valid = False
                    break
            if valid:
                fixed_frame = i
                break

    if fixed_frame:
        prev_dist = prev_pos = prev_len = None
        first_ball = False
        last_hip_y_pix = None
        for n in range(fixed_frame, total_frames):
            frame = frames[n]

            lm = landmarks_list[n]
            if not lm: continue

            elbow = (int(lm[RIGHT_ELBOW].x * width), int(lm[RIGHT_ELBOW].y * height))
            wrist = (int(lm[RIGHT_WRIST].x * width), int(lm[RIGHT_WRIST].y * height))
            index = (int(lm[RIGHT_INDEX].x * width), int(lm[RIGHT_INDEX].y * height))
            hand = ((wrist[0] + index[0]) // 2, (wrist[1] + index[1]) // 2)
            arm_len = np.linalg.norm(np.array(elbow) - np.array(wrist))

            if arm_len < 20 or (prev_len and arm_len < prev_len * 0.2):
                continue

            lh, rh = lm[LEFT_HIP], lm[RIGHT_HIP]
            vis_ok = (getattr(lh, "visibility", 1.0) > 0.5 and getattr(rh, "visibility", 1.0) > 0.5)
            if vis_ok:
                hip_y_pix = int(((lh.y + rh.y) / 2.0) * height)
                last_hip_y_pix = hip_y_pix
            else:
                hip_y_pix = last_hip_y_pix

            WAIST_MARGIN = max(4, int(0.01 * height))

            results = yolo_model.predict(source=frame, conf=0.2, verbose=False)
            candidates = []
            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None: continue
                for box in r.boxes:
                    try:
                        score = float(box.conf.cpu().item())
                    except Exception:
                        score = float(box.conf)
                    if score < 0.10: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if hip_y_pix is not None and cy >= hip_y_pix - WAIST_MARGIN: continue
                    d2 = (cx - hand[0]) ** 2 + (cy - hand[1]) ** 2
                    candidates.append((d2, (cx, cy)))

            detected, ball_pos = False, None
            if candidates:
                candidates.sort(key=lambda t: t[0])
                ball_pos = candidates[0][1]
                detected = True

            if detected and not first_ball:
                first_ball = True
                prev_pos = ball_pos
            elif not detected:
                if not first_ball: continue
                ball_pos = prev_pos

            if ball_pos is not None:
                dist = np.linalg.norm(np.array(hand) - np.array(ball_pos))
                if prev_dist and abs(dist - prev_dist) > 50: continue
                if prev_pos and np.linalg.norm(np.array(prev_pos) - np.array(ball_pos)) < 2: continue
                if dist > arm_len * 0.5:
                    release_frame = n
                    break
                prev_dist, prev_pos, prev_len = dist, ball_pos, arm_len

    # Define follow_frame based on release_frame, if available
    follow_frame = (release_frame + 15) if release_frame is not None else None

    frame_list = [start_frame, max_knee_frame, fixed_frame, release_frame, follow_frame]

    # Calculate shin_length_pixels for metric conversion
    shin_length_pixels = 0
    if max_knee_frame is not None and landmarks_list[max_knee_frame]:
        lm = landmarks_list[max_knee_frame]
        # Use LEFT_KNEE and LEFT_ANKLE for shin length calculation
        if LEFT_KNEE < len(lm) and LEFT_ANKLE < len(lm):
            knee_coords = np.array([lm[LEFT_KNEE].x * width, lm[LEFT_KNEE].y * height])
            ankle_coords = np.array([lm[LEFT_ANKLE].x * width, lm[LEFT_ANKLE].y * height])
            shin_length_pixels = np.linalg.norm(knee_coords - ankle_coords)

    m_per_px = SHIN_LENGTH_M / shin_length_pixels if shin_length_pixels > 0 else 0.0

    # Compute wrist speed (from fixed_frame to follow_frame)
    wrist_speeds_mps = np.full(len(landmarks_list), np.nan)
    if fixed_frame is not None and follow_frame is not None:
        wrist_speeds_pxps = compute_wrist_speed_series_smoothed(
            landmarks_list=landmarks_list,
            width=width,
            height=height,
            fps=fps,
            start_frame=fixed_frame,
            end_frame=follow_frame,
            wrist_idx=RIGHT_WRIST
        )
        wrist_speeds_mps = wrist_speeds_pxps * m_per_px

    # Compute shoulder angular velocity (from fixed_frame to follow_frame)
    shoulder_angular_velocities_degps = np.full(len(landmarks_list), np.nan)
    if fixed_frame is not None and follow_frame is not None:
        _, shoulder_angular_velocities_degps = compute_shoulder_angular_velocity_series(
            landmarks_list=landmarks_list,
            width=width,
            height=height,
            fps=fps,
            start_frame=fixed_frame,
            end_frame=follow_frame,
            side='right'  # Assuming right-handed pitcher
        )

    # Compute arm trajectory (from fixed_frame to follow_frame)
    arm_trajectory = []
    if fixed_frame is not None and follow_frame is not None:
        arm_trajectory = get_arm_trajectory(
            landmarks_list,
            width,
            height,
            start_frame=fixed_frame,
            end_frame=follow_frame,
            wrist_idx=RIGHT_WRIST
        )

    return {
        'frame_list': frame_list,
        'fixed_frame': fixed_frame,
        'release_frame': release_frame,
        'frames': frames,
        'landmarks_list': landmarks_list,
        'width': width,
        'height': height,
        'max_knee_frame': max_knee_frame,
        'wrist_speeds_mps': wrist_speeds_mps,  # Added wrist speeds
        'shoulder_angular_velocities_degps': shoulder_angular_velocities_degps,  # Added shoulder angular velocities
        'arm_trajectory': arm_trajectory,  # Added arm trajectory
        'fps': fps
    }


# =================================================================================
# Calculation & Metric Functions
# =================================================================================

def get_ball_trajectory_and_speed(video_path, release_frame, yolo_model, shin_length_pixels, SHIN_LENGTH_M=0.45,
                                  VIDEO_FPS=120):
    """
    Calculates ball speed and trajectory after the release frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {"trajectory": [], "speed_kph": None}

    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)

    tracked_positions = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
        ball_found = False
        if results and hasattr(results[0], "boxes") and results[0].boxes:
            best_box = max(results[0].boxes, key=lambda box: box.conf)
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tracked_positions.append((cx, cy))
            ball_found = True
        if not ball_found: break
    cap.release()

    speed_kph = None
    if len(tracked_positions) >= 2 and shin_length_pixels > 0:
        first_pos = np.array(tracked_positions[0])
        last_pos = np.array(tracked_positions[-1])
        pixel_distance = np.linalg.norm(last_pos - first_pos)
        real_distance_m = (pixel_distance / shin_length_pixels) * SHIN_LENGTH_M
        time_s = (len(tracked_positions) - 1) / VIDEO_FPS
        if time_s > 0:
            speed_kph = (real_distance_m / time_s) * 3.6

    return {"trajectory": tracked_positions, "speed_kph": speed_kph}


def calculate_angle(a, b, c):
    """Calculates the angle between three points (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-9: return 0.0
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def get_joint_angles(lm, width, height):
    """
    Extracts key joint angles (arm, leg) and torso tilt from landmarks.
    """
    if lm is None: return {}
    P = lambda i: (int(lm[i].x * width), int(lm[i].y * height))

    # Right Arm Angle
    shoulder, elbow, wrist = P(RIGHT_SHOULDER), P(RIGHT_ELBOW), P(RIGHT_WRIST)
    arm_angle = calculate_angle(shoulder, elbow, wrist)

    # Left Leg Angle
    left_hip, left_knee, left_ankle = P(LEFT_HIP), P(LEFT_KNEE), P(LEFT_ANKLE)
    leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # Torso Tilt
    pelvis_center = ((P(LEFT_HIP)[0] + P(RIGHT_HIP)[0]) // 2, (P(LEFT_HIP)[1] + P(RIGHT_HIP)[1]) // 2)
    shoulder_center = ((P(LEFT_SHOULDER)[0] + P(RIGHT_SHOULDER)[0]) // 2,
                       (P(LEFT_SHOULDER)[1] + P(RIGHT_SHOULDER)[1]) // 2)
    vec = (pelvis_center[0] - shoulder_center[0], pelvis_center[1] - shoulder_center[1])
    tilt = math.degrees(math.atan2(vec[0], vec[1]))

    return {
        "arm_angle": arm_angle, "leg_angle": leg_angle, "tilt": tilt,
        "shoulder": shoulder, "elbow": elbow, "wrist": wrist,
        "left_hip": left_hip, "left_knee": left_knee, "left_ankle": left_ankle,
        "pelvis_center": pelvis_center, "shoulder_center": shoulder_center
    }


def get_hand_height(lm, width, height, SHIN_LENGTH_M=0.45):
    """
    Calculates the normalized and real-world height of the hand relative to the ankle.
    """
    if lm is None: return {}
    ankle = (int(lm[LEFT_ANKLE].x * width), int(lm[LEFT_ANKLE].y * height))
    wrist = (int(lm[RIGHT_WRIST].x * width), int(lm[RIGHT_WRIST].y * height))
    knee = (int(lm[LEFT_KNEE].x * width), int(lm[LEFT_KNEE].y * height))

    shin_len = np.linalg.norm(np.array(knee) - np.array(ankle))
    dy = ankle[1] - wrist[1]

    normalized_height = dy / shin_len if shin_len > 0 else None
    real_height = normalized_height * SHIN_LENGTH_M if normalized_height is not None else None

    return {
        "normalized_height": normalized_height, "real_height": real_height,
        "wrist": wrist, "ankle": ankle, "knee": knee
    }


def calculate_frame_by_frame_metrics(analysis_result: dict) -> List[Optional[Dict]]:
    """
    Calculates key metrics for each frame in the analysis result.
    """
    landmarks_list = analysis_result['landmarks_list']
    width = analysis_result['width']
    height = analysis_result['height']

    all_metrics = []

    for lm in landmarks_list:
        if lm is None:
            all_metrics.append(None)
            continue

        try:
            angles = get_joint_angles(lm, width, height)
            height_info = get_hand_height(lm, width, height)

            frame_metrics = {
                'torso_tilt': angles.get('tilt'),
                'elbow_angle': angles.get('arm_angle'),
                'knee_angle': angles.get('leg_angle'),
                'hand_height_m': height_info.get('real_height')
            }
            all_metrics.append(frame_metrics)
        except Exception:
            all_metrics.append(None)

    return all_metrics


def compute_wrist_speed_series_smoothed(landmarks_list, width, height, fps, start_frame, end_frame,
                                        window_radius=2, min_vis=0.5, wrist_idx=RIGHT_WRIST):
    """
    이동 평균 필터를 적용하여 노이즈가 적고 부드러운 손목 속도(px/s)를 계산합니다.
    """
    T = len(landmarks_list)
    speed_pxps = np.full(T, np.nan, dtype=np.float64)

    s_coord = max(0, start_frame - window_radius)
    e_coord = min(T, end_frame + window_radius + 1)

    coords = [get_xy_px(landmarks_list, t, wrist_idx, width, height, min_vis=min_vis) for t in range(s_coord, e_coord)]

    diffs = [np.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]) if coords[i] and coords[
        i + 1] else np.nan for i in range(len(coords) - 1)]

    dt = 1.0 / float(fps)
    s_speed = max(start_frame, window_radius)
    e_speed = min(end_frame, T - 1 - window_radius)

    for t in range(s_speed, e_speed + 1):
        start_diff_idx = (t - window_radius) - s_coord
        end_diff_idx = (t + window_radius) - s_coord

        segment = [diffs[i] for i in range(start_diff_idx, end_diff_idx) if
                   i >= 0 and i < len(diffs) and not np.isnan(diffs[i])]

        if segment:
            speed_pxps[t] = float(np.mean(segment)) / dt

    return speed_pxps


def compute_shoulder_angular_velocity_series(landmarks_list, width, height, fps, start_frame, end_frame,
                                             window_radius=2, min_vis=0.5, side='right'):
    idxs = (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HIP) if side == 'right' else (LEFT_SHOULDER, LEFT_HIP)
    T = len(landmarks_list)
    theta = np.full(T, np.nan)

    s0, e0 = max(0, start_frame - window_radius), min(T - 1, end_frame + window_radius)
    for t in range(s0, e0 + 1):
        S, E, H = [get_xy_px(landmarks_list, t, i, width, height, min_vis) for i in idxs]
        if S and E and H:
            theta[t] = _angle_between_2d((E[0] - S[0], E[1] - S[1]), (H[0] - S[0], H[1] - S[1]))

    valid_indices = ~np.isnan(theta)
    if np.any(valid_indices):
        theta[valid_indices] = np.unwrap(theta[valid_indices])

    omega = np.full(T, np.nan)
    dt = 1.0 / float(fps)
    s, e = max(start_frame, window_radius), min(end_frame, T - 1 - window_radius)

    for t in range(s, e + 1):
        vals = [(theta[t + j] - theta[t - j]) / (2.0 * j * dt) for j in range(1, window_radius + 1) if
                t + j < T and t - j >= 0 and not (np.isnan(theta[t + j]) or np.isnan(theta[t - j]))]
        if vals: omega[t] = np.mean(vals)

    return omega, omega * (180.0 / math.pi)


def get_arm_trajectory(landmarks_list: List[Optional[List[mp.solutions.pose.PoseLandmark]]],
                       # Changed type hint to List[mp.solutions.pose.PoseLandmark]
                       width: int, height: int,
                       start_frame: Optional[int], end_frame: Optional[int],
                       wrist_idx: int = RIGHT_WRIST,
                       min_vis: float = 0.5) -> List[Tuple[int, int]]:
    """
    Extracts the 2D pixel trajectory of a specified wrist landmark over a given frame range.
    """
    trajectory = []
    if start_frame is None or end_frame is None:
        return trajectory

    # Ensure frame range is within the bounds of landmarks_list
    actual_start_frame = max(0, start_frame)
    actual_end_frame = min(end_frame, len(landmarks_list) - 1)

    for i in range(actual_start_frame, actual_end_frame + 1):
        if landmarks_list[i] is not None:
            lm = landmarks_list[i]
            # Add a check to ensure wrist_idx is a valid index for lm
            if wrist_idx < len(lm):
                p = lm[wrist_idx]
                if getattr(p, "visibility", 1.0) >= min_vis:
                    trajectory.append((int(p.x * width), int(p.y * height)))
    return trajectory


# =================================================================================
# DTW and Similarity Functions
# =================================================================================

def extract_and_normalize_landmarks(landmarks_list, used_ids, width, height):
    """
    Extracts and normalizes specified landmarks relative to the hip center.
    """
    normalized_landmarks = []
    for lm in landmarks_list:
        if lm is None: continue
        center_x = (lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2
        center_y = (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2
        frame_coords = []
        for idx in used_ids:
            norm_x = lm[idx].x - center_x
            norm_y = lm[idx].y - center_y
            frame_coords.extend([norm_x, norm_y])
        normalized_landmarks.append(frame_coords)
    return np.array(normalized_landmarks)


def get_phase_ranges(frame_list):
    """Converts a list of key frames into a list of (start, end) tuples for each phase."""
    start, max_knee, fixed, release, follow = frame_list
    return [(start, max_knee), (max_knee, fixed), (fixed, release), (release, follow)]


def score_from_distance(dist, min_dist=0.0, max_dist=0.15):
    """Converts DTW distance to a similarity score from 0 to 100."""
    if dist <= min_dist: return 100.0
    if dist >= max_dist: return 0.0
    return 100.0 * (1 - (dist - min_dist) / (max_dist - min_dist))


def resample_to_reference_timeline(ref_seq: np.ndarray, test_seq: np.ndarray):
    """Aligns a test sequence to a reference sequence's timeline using FastDTW."""
    dtw_dist, path = fastdtw(ref_seq, test_seq, dist=euclidean)
    N, D = ref_seq.shape
    out, cnt = np.zeros((N, D), dtype=np.float32), np.zeros(N, dtype=np.int32)

    for i, j in path:
        out[i] += test_seq[j]
        cnt[i] += 1

    nz = cnt > 0
    out[nz] /= cnt[nz, None]

    for i in range(N):
        if cnt[i] == 0:
            L, R = i - 1, i + 1
            while L >= 0 and cnt[L] == 0: L -= 1
            while R < N and cnt[R] == 0: R += 1
            if L >= 0 and R < N:
                out[i] = 0.5 * (out[L] + out[R])
            elif L >= 0:
                out[i] = out[L]
            elif R < N:
                out[i] = out[R]

    per_frame_err = np.linalg.norm(ref_seq - out, axis=1) if len(out) else np.array([])
    return out, dtw_dist, path, per_frame_err


def evaluate_pair_with_dynamic_masks(reference_video: str, test_video: str, used_ids: List[int], yolo_model,
                                     min_phase_len: int = 5, strict: bool = True):
    """
    Compares two analyzed videos phase by phase using DTW with dynamic joint masking.
    """
    ref = analyze_video(reference_video, yolo_model)
    test = analyze_video(test_video, yolo_model)
    ok_ref, info_ref = quick_phase_report(reference_video, ref, min_phase_len=min_phase_len)
    ok_tst, info_tst = quick_phase_report(test_video, test, min_phase_len=min_phase_len)
    if strict and (not ok_ref or not ok_tst):
        raise PhaseSegmentationError(f"영상 분할 실패. REF: {info_ref.get('reason')} | TST: {info_tst.get('reason')}")

    p_ref = get_phase_ranges(ref['frame_list'])
    p_tst = get_phase_ranges(test['frame_list'])

    def run_dtw_phase(res_ref, res_tst, start_end_ref, start_end_tst, include_wr: bool, include_an: bool):
        ids_masked = masked_used_ids(used_ids, include_wr, include_an)
        seq_ref = extract_and_normalize_landmarks(res_ref['landmarks_list'], ids_masked, res_ref['width'],
                                                  res_ref['height'])
        seq_tst = extract_and_normalize_landmarks(res_tst['landmarks_list'], ids_masked, res_tst['width'],
                                                  res_tst['height'])

        s_ref, e_ref = start_end_ref
        s_tst, e_tst = start_end_tst

        if s_ref is None or e_ref is None or s_tst is None or e_tst is None:
            return float('inf'), 0

        seg_ref = seq_ref[s_ref:e_ref]
        seg_tst = seq_tst[s_tst:e_tst]

        if len(seg_ref) == 0 or len(seg_tst) == 0: return float('inf'), 0

        _, dtw_dist, path, _ = resample_to_reference_timeline(seg_ref, seg_tst)
        return float(dtw_dist) / max(1, len(path)), len(path)

    phase_scores, phase_costs = [], []
    masks = [(False, False), (False, False), (True, True), (True, True)]

    for i, (p_r, p_t) in enumerate(zip(p_ref, p_tst)):
        if p_r[0] is None or p_r[1] is None or p_t[0] is None or p_t[1] is None:
            score, cost = 0.0, float('inf')
        else:
            cost, _ = run_dtw_phase(ref, test, p_r, p_t, *masks[i])
            score = score_from_distance(cost) if math.isfinite(cost) else 0.0
        phase_scores.append(score)
        phase_costs.append(cost)

    overall_score = np.mean([s for s in phase_scores if math.isfinite(s)]) if any(
        math.isfinite(s) for s in phase_scores) else 0.0
    worst_phase_idx = np.argmin(phase_scores) + 1 if phase_scores and np.any(np.isfinite(phase_scores)) else None

    return phase_scores, phase_costs, overall_score, worst_phase_idx


# =================================================================================
# Utility and Helper Functions
# =================================================================================

def phase_sanity_check(frame_list, total_frames, min_phase_len=5):
    """
    analyze_video에서 찾은 주요 프레임 리스트가 유효한지 검사합니다.
    (None 값 여부, 시간순 정렬 여부, 최소 길이 충족 여부 등)
    """
    names = ["start", "max_knee", "fixed", "release", "follow"]
    info = {"frame_list": dict(zip(names, frame_list)), "lengths": {}}

    if any(f is None for f in frame_list):
        info["reason"] = "contains None"
        return False, info

    s, m, f, r, fo = frame_list

    if not (0 <= s < m < f < r < fo <= total_frames):
        info["reason"] = "non-monotonic or out-of-range"
        return False, info

    phases = [(s, m), (m, f), (f, r), (r, fo)]
    for idx, (a, b) in enumerate(phases, 1):
        info["lengths"][f"phase{idx}"] = b - a
        if b - a < min_phase_len:
            info["reason"] = f"phase{idx} too short (<{min_phase_len})"
            return False, info

    info["reason"] = "ok"
    return True, info


def quick_phase_report(video_path, analyze_result, min_phase_len=5):
    """
    phase_sanity_check를 호출하여 영상의 단계 분할 결과를 간단히 출력합니다.
    """
    total_frames = len(analyze_result['frames'])
    frame_list = analyze_result['frame_list']
    ok, info = phase_sanity_check(frame_list, total_frames, min_phase_len=min_phase_len)
    if ok:
        print(f"[분할 점검 OK] {video_path} | lengths: {info['lengths']}")
    else:
        print(f"[분할 점검 FAIL] {video_path} | reason: {info.get('reason')} | frames: {info['frame_list']}")
    return ok, info


def masked_used_ids(base_used_ids: List[int], include_wrists: bool, include_ankles: bool) -> List[int]:
    s = set(base_used_ids)
    if not include_wrists: s -= {LEFT_WRIST, RIGHT_WRIST}
    if not include_ankles: s -= {LEFT_ANKLE, RIGHT_ANKLE}
    return sorted(s)


def get_xy_px(lm_list, frame_idx, idx, width, height, min_vis=0.5):
    if not (0 <= frame_idx < len(lm_list)) or lm_list[frame_idx] is None: return None
    p = lm_list[frame_idx][idx]
    if getattr(p, "visibility", 1.0) < min_vis: return None
    return (float(p.x * width), float(p.y * height))


def _angle_between_2d(u, v):
    return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


def shin_len_px_at(lms, W, H, frame_idx):
    if isinstance(lms, list):
        a = get_xy_px(lms, frame_idx, LEFT_KNEE, W, H)
        b = get_xy_px(lms, frame_idx, LEFT_ANKLE, W, H)
    else:  # Handle single landmark object
        a = (lms[LEFT_KNEE].x * W, lms[LEFT_KNEE].y * H)
        b = (lms[LEFT_ANKLE].x * W, lms[LEFT_ANKLE].y * H)
    if a and b: return math.hypot(a[0] - b[0], a[1] - b[1])
    return None


# =================================================================================
# Video and Image Rendering Functions
# =================================================================================

def build_percentile_range_abs(arr, lo=5, hi=95):
    v = arr.copy()[~np.isnan(arr)]
    v = np.abs(v)
    if v.size == 0: return 0.0, 1.0
    a, b = np.percentile(v, [lo, hi])
    if abs(b - a) < 1e-9: a, b = float(v.min()), float(v.max() + 1e-6)
    return float(a), float(b)


def speed_to_color_bgr(v_mps, v_low, v_high):
    if v_mps is None or np.isnan(v_mps): return (0, 255, 255)
    vv = float(max(v_low, min(abs(v_mps), v_high)))
    t = (vv - v_low) / (v_high - v_low + 1e-9)
    g, r = int(round(255 * (1.0 - t))), int(round(255 * t))
    return (0, g, r)


def omega_to_color_bgr(w_degps, w_low, w_high):
    if w_degps is None or np.isnan(w_degps): return (0, 255, 255)
    vv = float(max(w_low, min(abs(w_degps), w_high)))
    t = (vv - w_low) / (w_high - w_low + 1e-9)
    g, r = int(round(255 * (1.0 - t))), int(round(255 * t))
    return (0, g, r)


def draw_color_legend(img, lo_val, hi_val, label="Value", pos=(20, 20), size=(220, 20), font_scale=0.6, thick=1):
    x0, y0 = pos
    w, h = size
    for i in range(w):
        t = i / (w - 1)
        g, r = int(round(255 * (1.0 - t))), int(round(255 * t))
        cv2.line(img, (x0 + i, y0), (x0 + i, y0 + h), (0, g, r), 1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), 1)
    cv2.putText(img, f"{lo_val:.1f}", (x0, y0 + h + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick,
                cv2.LINE_AA)
    cv2.putText(img, f"{hi_val:.1f}", (x0 + w - 36, y0 + h + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                thick, cv2.LINE_AA)
    cv2.putText(img, label, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick, cv2.LINE_AA)


def draw_speed_label_box(img, value, value_color_bgr, label="Speed", unit=" m/s", pad=12, thick=2, font_scale=1.15):
    H, W = img.shape[:2]
    disp_v = 0.0 if (value is None or np.isnan(float(value))) else float(value)
    left_str, mid_str, right_str = f"{label}", f" {disp_v:.2f}", f"{unit}"
    full_str = left_str + mid_str + right_str
    (tw_total, th), _ = cv2.getTextSize(full_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    bw, bh = tw_total + 2 * pad, th + 2 * pad
    x, y = W - bw - 20, 20
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 255, 255), -1)
    x0, y0 = x + pad, y + th + pad - 2
    (tw_left, _), _ = cv2.getTextSize(left_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    (tw_mid, _), _ = cv2.getTextSize(mid_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    cv2.putText(img, left_str, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, mid_str, (x0 + tw_left, y0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, value_color_bgr, thick,
                cv2.LINE_AA)
    cv2.putText(img, right_str, (x0 + tw_left + tw_mid, y0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick,
                cv2.LINE_AA)


def _get_video_writer(save_path: str, fps: int, frame_size: Tuple[int, int]):
    """
    Tries to find and return a working cv2.VideoWriter instance.
    It attempts a list of common FOURCC codecs for MP4 files.
    """
    codecs = ['avc1', 'mp4v']  # H.264 is generally preferred and more modern
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
            if writer.isOpened():
                print(f"VideoWriter opened successfully with codec '{codec}'.")
                return writer
        except Exception as e:
            print(f"Codec '{codec}' failed to initialize: {e}")

    print(f"ERROR: Could not open VideoWriter for path '{save_path}' with any of the attempted codecs: {codecs}.")
    return None


def draw_bones_diamond_batched(img, segments, color_main, width=12):
    """
    Draws thick, diamond-shaped lines for bones between two points.
    """
    if not segments:
        return
    off = width * 0.5
    for p1, p2 in segments:
        (x1, y1), (x2, y2) = p1, p2
        vx, vy = (x2 - x1, y2 - y1)
        L = math.hypot(vx, vy)
        if L < 1e-6:
            continue
        nx, ny = (-vy / L, vx / L)
        mx, my = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        p_top = (int(round(mx + nx * off)), int(round(my + ny * off)))
        p_bottom = (int(round(mx - nx * off)), int(round(my - ny * off)))
        poly = np.array([[x1, y1], [p_top[0], p_top[1]],
                         [x2, y2], [p_bottom[0], p_bottom[1]]], np.int32)
        cv2.fillConvexPoly(img, poly, color_main)


def _draw_skeleton_for_frame(frame, lm, W, H, used_ids, current_phase_idx):
    if not lm:
        return

    # 왼쪽 팔꿈치(13)와 왼쪽 손목(15)을 렌더링에서 제외합니다.
    # LEFT_ELBOW = 13, LEFT_WRIST = 15
    used_ids = [i for i in used_ids if i not in [13, 15]]

    visible = set(used_ids)
    if current_phase_idx in (1, 2):
        visible -= {LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE}

    for i in visible:
        if 0 <= i < len(lm):
            x, y = int(lm[i].x * W), int(lm[i].y * H)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, (0, 0, 0), 2, cv2.LINE_AA)

    def segs(edges):
        out_segs = []
        for a, b in edges:
            if a in visible and b in visible and 0 <= a < len(lm) and 0 <= b < len(lm):
                ax, ay = int(lm[a].x * W), int(lm[a].y * H)
                bx, by = int(lm[b].x * W), int(lm[b].y * H)
                out_segs.append(((ax, ay), (bx, by)))
        return out_segs

    draw_bones_diamond_batched(frame, segs(LEFT_EDGES), (0, 255, 255), 12)
    draw_bones_diamond_batched(frame, segs(RIGHT_EDGES), (0, 255, 0), 12)
    draw_bones_diamond_batched(frame, segs(CENTER_EDGES), (238, 238, 238), 12)


def render_skeleton_video(analysis_result: dict, save_path: str, used_ids: List[int], fps: int = 30):
    """
    Renders a skeleton overlay video from analysis results and saves it to a file.
    """
    frames = analysis_result['frames']
    lms_list = analysis_result['landmarks_list']
    W, H = analysis_result['width'], analysis_result['height']
    start, max_knee, fixed, release, follow = analysis_result['frame_list']

    if start is None or follow is None:
        print("Warning: Cannot render skeleton video due to incomplete phase segmentation.")
        return None

    phase_ranges = [
        (start, max_knee, "Phase 1: Windup"),
        (max_knee, fixed, "Phase 2: Stride"),
        (fixed, release, "Phase 3: Release"),
        (release, follow, "Phase 4: Follow-through")
    ]

    out = _get_video_writer(save_path, fps, (W, H))

    if out is None:
        return None

    if not out.isOpened():
        print(f"Error: Could not open video writer for path {save_path}")
        return None


    for abs_t in range(start, follow):
        current_phase_label = ""
        current_phase_idx = 0
        for i, (ps, pe, label) in enumerate(phase_ranges, 1):
            if ps is not None and pe is not None and ps <= abs_t < pe:
                current_phase_label = label
                current_phase_idx = i
                break

        frame = frames[abs_t].copy()
        lm = lms_list[abs_t]

        _draw_skeleton_for_frame(frame, lm, W, H, used_ids, current_phase_idx)

        if current_phase_label:
            cv2.rectangle(frame, (20, 20), (450, 80), (255, 255, 255), -1)
            cv2.putText(frame, current_phase_label, (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

        out.write(frame)

    out.release()
    return save_path


def render_arm_swing_speed_video(analysis_result: dict, save_path: str, used_ids: List[int], fps: int = 30):
    """
    Renders a video visualizing the arm swing speed trajectory with skeleton.
    """
    frames = analysis_result['frames']
    lms_list = analysis_result['landmarks_list']
    W, H = analysis_result['width'], analysis_result['height']
    start, max_knee, fixed, release, follow = analysis_result['frame_list']
    v_mps = analysis_result['wrist_speeds_mps']

    if start is None or follow is None or fixed is None:
        print("Warning: Cannot render arm swing speed video due to incomplete phase segmentation.")
        return None

    out = _get_video_writer(save_path, fps, (W, H))
    if not out or not out.isOpened():
        print(f"Error: Could not open video writer for path {save_path}")
        return None

    phase_ranges = [
        (start, max_knee), (max_knee, fixed),
        (fixed, release), (release, follow)
    ]

    w_s = fixed
    w_e = min(follow, len(v_mps) - 1)
    v_low, v_high = build_percentile_range_abs(v_mps[w_s:w_e + 1])

    trail_layer = np.zeros((H, W, 3), dtype=np.uint8)
    TRAIL_THICKNESS, TRAIL_ALPHA = 16, 0.85
    prev_wrist_px = None


    for frame_idx in range(start, follow):
        frame = frames[frame_idx].copy()  # Original video background
        lm = lms_list[frame_idx]

        # --- Speed-based trail rendering ---
        if fixed <= frame_idx < follow:
            v_curr = v_mps[frame_idx]
            if not np.isnan(v_curr):
                col = speed_to_color_bgr(v_curr, v_low, v_high)
                p_wrist = get_xy_px(lms_list, frame_idx, RIGHT_WRIST, W, H)

                if p_wrist:
                    wx, wy = int(p_wrist[0]), int(p_wrist[1])
                    if prev_wrist_px is not None:
                        cv2.line(trail_layer, prev_wrist_px, (wx, wy), col, TRAIL_THICKNESS, cv2.LINE_AA)
                    prev_wrist_px = (wx, wy)
                else:
                    prev_wrist_px = None

        # --- Skeleton Drawing (after trail calculation, before composition) ---
        if lm:
            current_phase_idx = 0
            for i, (ps, pe) in enumerate(phase_ranges, 1):
                if ps is not None and pe is not None and ps <= frame_idx < pe:
                    current_phase_idx = i
                    break
            # Draw skeleton on the original frame, not the final composed one
            _draw_skeleton_for_frame(frame, lm, W, H, used_ids, current_phase_idx)

        final_frame = cv2.addWeighted(frame, 1.0, trail_layer, TRAIL_ALPHA, 0)

        # --- Labels and Legends ---
        if fixed <= frame_idx < follow:
            v_curr = v_mps[frame_idx]
            if not np.isnan(v_curr):
                col = speed_to_color_bgr(v_curr, v_low, v_high)
                draw_speed_label_box(final_frame, v_curr, col, label="Wrist Speed", unit=" m/s")

        draw_color_legend(final_frame, v_low, v_high, label="Wrist v (m/s)", pos=(20, 20))

        out.write(final_frame)

    out.release()
    return save_path


def render_shoulder_angular_velocity_video(analysis_result: dict, save_path: str, used_ids: List[int], fps: int = 30):
    """
    Renders a video visualizing the shoulder angular velocity with skeleton.
    """
    frames = analysis_result['frames']
    lms_list = analysis_result['landmarks_list']
    W, H = analysis_result['width'], analysis_result['height']
    start, max_knee, fixed, release, follow = analysis_result['frame_list']
    omega_degps = analysis_result['shoulder_angular_velocities_degps']

    if start is None or follow is None or fixed is None:
        print("Warning: Cannot render shoulder angular velocity video due to incomplete phase segmentation.")
        return None

    out = _get_video_writer(save_path, fps, (W, H))
    if not out or not out.isOpened():
        print(f"Error: Could not open video writer for path {save_path}")
        return save_path

    phase_ranges = [(start, max_knee), (max_knee, fixed), (fixed, release), (release, follow)]

    w_s = fixed
    w_e = min(follow, len(omega_degps) - 1)
    w_low, w_high = build_percentile_range_abs(omega_degps[w_s:w_e + 1])

    trail_layer = np.zeros((H, W, 3), dtype=np.uint8)
    TRAIL_ALPHA = 0.85


    for frame_idx in range(start, follow):
        frame = frames[frame_idx].copy()  # Original video background
        lm = lms_list[frame_idx]

        # --- Angular velocity visualization ---
        if fixed <= frame_idx < follow:
            w_curr = omega_degps[frame_idx]
            if not np.isnan(w_curr):
                col = omega_to_color_bgr(w_curr, w_low, w_high)
                p_shoulder = get_xy_px(lms_list, frame_idx, RIGHT_SHOULDER, W, H)
                p_elbow = get_xy_px(lms_list, frame_idx, RIGHT_ELBOW, W, H)

                if p_shoulder and p_elbow:
                    sx, sy = int(p_shoulder[0]), int(p_shoulder[1])
                    ex, ey = int(p_elbow[0]), int(p_elbow[1])
                    # Draw trajectory on trail_layer
                    cv2.line(trail_layer, (sx, sy), (ex, ey), col, 14, cv2.LINE_AA)

        # --- Skeleton Drawing (after trail calculation, before composition) ---
        if lm:
            current_phase_idx = 0
            for i, (ps, pe) in enumerate(phase_ranges, 1):
                if ps is not None and pe is not None and ps <= frame_idx < pe:
                    current_phase_idx = i
                    break
            # Draw skeleton on the original frame
            _draw_skeleton_for_frame(frame, lm, W, H, used_ids, current_phase_idx)

        # --- Composition and Labels ---
        final_frame = cv2.addWeighted(frame, 1.0, trail_layer, TRAIL_ALPHA, 0)

        if fixed <= frame_idx < follow:
            w_curr = omega_degps[frame_idx]
            if not np.isnan(w_curr):
                col = omega_to_color_bgr(w_curr, w_low, w_high)
                draw_speed_label_box(final_frame, w_curr, col, label="Shoulder Speed", unit=" deg/s")

        draw_color_legend(final_frame, w_low, w_high, label="Shoulder Speed (deg/s)", pos=(20, 20))

        out.write(final_frame)

    out.release()
    return save_path


def render_ball_trajectory_video(analysis_result: dict, yolo_model, save_path: str, used_ids: List[int], fps: int = 30):
    """
    Renders a video visualizing the ball trajectory after release, with skeleton.
    The video covers the full range from start to follow-through.
    """
    frames = analysis_result['frames']
    lms_list = analysis_result['landmarks_list']
    W, H = analysis_result['width'], analysis_result['height']
    start, max_knee, fixed, release, follow = analysis_result['frame_list']

    if start is None or release is None or follow is None:
        print("Warning: Cannot render ball trajectory video without full phase segmentation.")
        return None

    out = _get_video_writer(save_path, fps, (W, H))
    if not out or not out.isOpened():
        print(f"Error: Could not open video writer for path {save_path}")
        return None

    phase_ranges = [(start, max_knee), (max_knee, fixed), (fixed, release), (release, follow)]
    ball_trail = np.zeros((H, W, 3), np.uint8)
    BALL_TRAIL_ALPHA = 0.9

    for frame_idx in range(start, follow):
        frame = frames[frame_idx].copy()  # Use original video frame
        lm = lms_list[frame_idx]

        # --- 릴리스 프레임에 분석 정보 오버레이 ---
        if frame_idx == release and lm is not None:
            # print(f"릴리스 프레임({frame_idx})에 분석 정보를 덧씌웁니다.")
            obstacles = []
            placed_rects = []

            # 계산
            angles = get_joint_angles(lm, W, H)
            height_info = get_hand_height(lm, W, H)

            # 선 그리기
            draw_hand_L_path(frame, height_info['ankle'], height_info['wrist'], (0, 0, 255), obstacles)
            draw_angle_lines(frame, angles['shoulder'], angles['elbow'], angles['wrist'], (0, 165, 255), obstacles)
            draw_angle_lines(frame, angles['left_hip'], angles['left_knee'], angles['left_ankle'], (0, 255, 0),
                             obstacles)
            draw_torso_line(frame, angles['shoulder_center'], angles['pelvis_center'], (255, 0, 255), 6, obstacles)

            # 텍스트 라벨(콜아웃) 그리기
            side_elbow = 'right' if angles['elbow'][0] < W * 0.5 else 'left'
            side_knee = 'right' if angles['left_knee'][0] < W * 0.5 else 'left'
            side_shoulder = 'right' if angles['shoulder_center'][0] < W * 0.5 else 'left'
            side_wrist = 'right' if angles['wrist'][0] < W * 0.5 else 'left'

            draw_callout_elbow_avoid_axes_pref(
                frame, angles['elbow'], f"Right Arm: {angles['arm_angle']:.1f} deg",
                side=side_elbow, tcolor=(0, 165, 255), placed_rects=placed_rects, avoid_rects=obstacles)

            draw_callout_elbow_avoid_axes_pref(
                frame, angles['left_knee'], f"Left Knee: {angles['leg_angle']:.1f} deg",
                side=side_knee, tcolor=(0, 255, 0), placed_rects=placed_rects, avoid_rects=obstacles)

            draw_callout_elbow_avoid_axes_pref(
                frame, angles['shoulder_center'], f"Tilt: {angles['tilt']:.1f} deg",
                side=side_shoulder, tcolor=(255, 0, 255), placed_rects=placed_rects, avoid_rects=obstacles)

            if height_info.get('real_height') is not None:
                draw_callout_elbow_avoid_axes_pref(
                    frame, angles['wrist'], f"Hand: {height_info['real_height']:.2f} m",
                    side=side_wrist, tcolor=(0, 0, 255), placed_rects=placed_rects, avoid_rects=obstacles)

        # --- Ball Detection and Trajectory (only from release frame onwards) ---
        if frame_idx >= release:
            yolo_out = yolo_model.predict(source=frame, conf=0.25, verbose=False, max_det=1)
            if yolo_out and yolo_out[0].boxes:
                best = yolo_out[0].boxes[0]
                x1, y1, x2, y2 = map(int, best.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3, cv2.LINE_AA)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(ball_trail, (cx, cy), 5, (255, 255, 0), -1, cv2.LINE_AA)

        # --- Skeleton Drawing (after overlays, before final composition) ---
        if lm:
            current_phase_idx = 0
            for i, (ps, pe) in enumerate(phase_ranges, 1):
                if ps is not None and pe is not None and ps <= frame_idx < pe:
                    current_phase_idx = i
                    break
            # Draw skeleton on the frame that might have release info
            _draw_skeleton_for_frame(frame, lm, W, H, used_ids, current_phase_idx)

        final_frame = cv2.addWeighted(frame, 1.0, ball_trail, BALL_TRAIL_ALPHA, 0)
        out.write(final_frame)

    out.release()
    return save_path


def render_release_allinone(result, save_path, shin_length_m=0.45):
    """
    Generates a single image of the release frame with key metrics overlaid.
    This is a utility function for creating visual reports.
    """
    rf = result['release_frame']
    if rf is None: raise RuntimeError("릴리스 프레임을 찾지 못했습니다.")

    frame = result['frames'][rf].copy()
    lm = result['landmarks_list'][rf]
    W, H = result['width'], result['height']

    if lm is None: raise RuntimeError("릴리스 프레임에서 포즈 랜드마크가 없습니다.")

    obstacles = []
    placed_rects = []

    P = lambda i: (int(lm[i].x * W), int(lm[i].y * H))

    # --- Calculations ---
    angles = get_joint_angles(lm, W, H)
    height_info = get_hand_height(lm, W, H, shin_length_m)

    # --- Drawing ---
    draw_hand_L_path(frame, height_info['ankle'], height_info['wrist'], (0, 0, 255), obstacles)
    draw_angle_lines(frame, angles['shoulder'], angles['elbow'], angles['wrist'], (0, 165, 255), obstacles)
    draw_angle_lines(frame, angles['left_hip'], angles['left_knee'], angles['left_ankle'], (0, 255, 0), obstacles)
    draw_torso_line(frame, angles['shoulder_center'], angles['pelvis_center'], (255, 0, 255), 6, obstacles)

    # --- Callouts ---
    side_elbow = 'right' if angles['elbow'][0] < W * 0.5 else 'left'
    side_knee = 'right' if angles['left_knee'][0] < W * 0.5 else 'left'
    side_shoulder = 'right' if angles['shoulder_center'][0] < W * 0.5 else 'left'
    side_wrist = 'right' if angles['wrist'][0] < W * 0.5 else 'left'

    draw_callout_elbow_avoid_axes_pref(
        frame, angles['elbow'], f"Right Arm: {angles['arm_angle']:.1f} deg",
        side=side_elbow, tcolor=(0, 165, 255),
        placed_rects=placed_rects, avoid_rects=obstacles,
        base_ext=260, max_ext=660, gap=12)

    draw_callout_elbow_avoid_axes_pref(
        frame, angles['left_knee'], f"Left Knee: {angles['leg_angle']:.1f} deg",
        side=side_knee, tcolor=(0, 255, 0),
        placed_rects=placed_rects, avoid_rects=obstacles,
        base_ext=260, max_ext=660, gap=12)

    draw_callout_elbow_avoid_axes_pref(
        frame, angles['shoulder_center'], f"Tilt: {angles['tilt']:.1f} deg",
        side=side_shoulder, tcolor=(255, 0, 255),
        placed_rects=placed_rects, avoid_rects=obstacles,
        base_ext=280, max_ext=700, gap=12)

    if height_info['real_height'] is not None:
        draw_callout_elbow_avoid_axes_pref(
            frame, angles['wrist'], f"Hand: {height_info['real_height']:.2f} m",
            side=side_wrist, tcolor=(0, 0, 255),
            placed_rects=placed_rects, avoid_rects=obstacles,
            base_ext=280, max_ext=700, gap=12)

    cv2.circle(frame, height_info['ankle'], 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, angles['wrist'], 7, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.imwrite(save_path, frame)

    return {
        "release_frame": rf,
        "right_arm_angle_deg": round(angles['arm_angle'], 1),
        "left_knee_angle_deg": round(angles['leg_angle'], 1),
        "torso_tilt_deg": round(angles['tilt'], 1),
        "hand_height_m": None if height_info['real_height'] is None else round(float(height_info['real_height']), 3),
        "saved_image_path": save_path
    }


def _rects_overlap(a, b, margin=6):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1 -= margin
    ay1 -= margin
    ax2 += margin
    ay2 += margin
    bx1 -= margin
    by1 -= margin
    bx2 += margin
    by2 += margin
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _line_intersects_rect(p1, p2, rect, margin=4):
    x1, y1, x2, y2 = rect
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    if x1 <= p1[0] <= x2 and y1 <= p1[1] <= y2: return True
    if x1 <= p2[0] <= x2 and y1 <= p2[1] <= y2: return True
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    ccw = lambda p, q, r: (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
    _seg_intersect = lambda a1, a2, b1, b2: (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (
                ccw(a1, a2, b1) != ccw(a1, a2, b2))
    return any(_seg_intersect(p1, p2, q1, q2) for q1, q2 in edges)


def _line_rect(p1, p2, margin=6):
    x1, y1 = p1
    x2, y2 = p2
    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
    return (x_min - margin, y_min - margin, x_max + margin, y_max + margin)


def _circle_rect(center, r, margin=4):
    x, y = center
    rr = r + margin
    return (x - rr, y - rr, x + rr, y + rr)


def draw_callout_elbow_avoid_axes_pref(
        img, anchor, text, side='right',
        bend=12, base_ext=240, max_ext=640, step=40,
        pad_axis=6, color=(255, 255, 255), tcolor=(0, 0, 0),
        thick=2, font_scale=0.95, pad=10, gap=10,
        placed_rects=None, avoid_rects=None
):
    if placed_rects is None: placed_rects = []
    if avoid_rects is None: avoid_rects = []
    H, W = img.shape[:2]
    ax, ay = anchor
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)

    def make_layout(ext, vertical='above'):
        sx = +1 if side == 'right' else -1
        yoff = -bend if vertical == 'above' else +bend
        p1 = (ax + sx * bend, ay + yoff)
        p2 = (p1[0] + sx * ext, p1[1])
        if side == 'right':
            box_x = p2[0] + gap
        else:
            box_x = p2[0] - (tw + 2 * pad) - gap
        if vertical == 'above':
            box_y = max(10, p2[1] - th - pad - 2)
        else:
            box_y = min(H - (th + 2 * pad) - 10, p2[1] + 8)
        rect = (box_x, box_y, box_x + tw + 2 * pad, box_y + th + 2 * pad)
        return p1, p2, rect

    def ok(p1, p2, rect):
        x1, y1, x2r, y2r = rect
        if x1 < 4 or x2r > W - 4 or y1 < 4 or y2r > H - 4: return False
        for r in placed_rects + avoid_rects:
            if _rects_overlap(rect, r): return False
            if _line_intersects_rect((ax, ay), p1, r): return False
            if _line_intersects_rect(p1, p2, r): return False
        return True

    chosen = None
    for vertical in ('above', 'below'):
        for ext in range(base_ext, max_ext + 1, step):
            p1, p2, rect = make_layout(ext, vertical)
            if ok(p1, p2, rect):
                chosen = (p1, p2, rect)
                break
        if chosen: break
    if not chosen:
        p1, p2, rect = make_layout(base_ext, 'above')
    else:
        p1, p2, rect = chosen

    cv2.line(img, (ax, ay), p1, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.line(img, p1, p2, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
    cv2.putText(img, text, (rect[0] + pad, rect[1] + th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, tcolor, thick, cv2.LINE_AA)

    placed_rects.append(rect)
    placed_rects += [_line_rect((ax, ay), p1, 6), _line_rect(p1, p2, 6)]
    return rect


def draw_angle_lines(img, a, b, c, color, obstacles):
    for p in [a, b, c]: cv2.circle(img, p, 6, color, -1, cv2.LINE_AA)
    cv2.line(img, a, b, color, 3, cv2.LINE_AA)
    cv2.line(img, c, b, color, 3, cv2.LINE_AA)
    obstacles += [_line_rect(a, b, 8), _line_rect(c, b, 8),
                  _circle_rect(a, 6), _circle_rect(b, 6), _circle_rect(c, 6)]


def draw_torso_line(img, shoulder, pelvis, color, thick, obstacles):
    cv2.line(img, shoulder, pelvis, color, thick, cv2.LINE_AA)
    for p in [shoulder, pelvis]: cv2.circle(img, p, 7, color, -1, cv2.LINE_AA)
    obstacles += [_line_rect(shoulder, pelvis, 10),
                  _circle_rect(shoulder, 7), _circle_rect(pelvis, 7)]


def draw_hand_L_path(img, ankle, wrist, color, obstacles):
    mid = (wrist[0], ankle[1])
    cv2.line(img, ankle, mid, color, 3, cv2.LINE_AA)
    cv2.line(img, mid, wrist, color, 3, cv2.LINE_AA)
    for p in [ankle, wrist]: cv2.circle(img, p, 7, color, -1, cv2.LINE_AA)
    obstacles += [_line_rect(ankle, mid, 8), _line_rect(mid, wrist, 8),
                  _circle_rect(ankle, 7), _circle_rect(wrist, 7)]
    return mid
