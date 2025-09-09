# Copyright (c) 2024 TACTICS. All Rights Reserved.
#
# 이 소프트웨어의 상업적 사용, 수정 및 배포를 금지합니다.
# 허가 없이 이 코드를 사용, 복제, 수정, 배포할 수 없습니다.
#
# Commercial use, modification, and distribution of this software are prohibited.
# You may not use, copy, modify, or distribute this code without permission.

import cv2
import numpy as np
import math
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os

# --- 1. 투구 동작 자동 분할 ---
def analyze_video(video_path, yolo_model=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2,
                                  smooth_landmarks=True, min_detection_confidence=0.7,
                                  min_tracking_confidence=0.5)

    frames, landmarks_list = [], []
    knee_y_list, ankle_y_list, ankle_x_list = [], [], []
    foot_x_list, foot_y_list = [], []

    LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX = 25, 27, 31
    RIGHT_KNEE, RIGHT_ANKLE = 26, 28
    SHIN_SCALE_DIVISOR = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
    print(f"total fremes: {total_frames}")
    print(f"landmarks: {len(landmarks_list)}")
    valid_knees = [(i, y) for i, y in enumerate(knee_y_list) if y is not None]
    max_knee_frame = min(valid_knees, key=lambda x: x[1])[0] if valid_knees else None

    threshold = None
    if max_knee_frame is not None:
        lm = landmarks_list[max_knee_frame]
        rk, ra = lm[RIGHT_KNEE], lm[RIGHT_ANKLE]
        x1, y1 = rk.x * width, rk.y * height
        x2, y2 = ra.x * width, ra.y * height
        threshold = math.hypot(x2 - x1, y2 - y1) / SHIN_SCALE_DIVISOR

    start_frame, fixed_frame, release_frame = None, None, None

    if threshold:
        count = 0
        for i in range(max_knee_frame - 5):
            if ankle_y_list[i] and ankle_y_list[i + 5]:
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
                if None in [
                    ankle_x_list[f1], ankle_x_list[f2], ankle_y_list[f1], ankle_y_list[f2],
                    foot_x_list[f1], foot_x_list[f2], foot_y_list[f1], foot_y_list[f2]
                ]:
                    valid = False
                    break
                if (
                    abs(ankle_x_list[f2] - ankle_x_list[f1]) > epsilon or
                    abs(ankle_y_list[f2] - ankle_y_list[f1]) > epsilon or
                    abs(foot_x_list[f2] - foot_x_list[f1]) > epsilon or
                    abs(foot_y_list[f2] - foot_y_list[f1]) > epsilon
                ):
                    valid = False
                    break
            if valid:
                fixed_frame = i
                break

    release_fail_reason = None
    if fixed_frame:
        prev_dist = prev_pos = prev_len = None
        first_ball = False
        for n in range(fixed_frame, total_frames):
            frame = frames[n]
            lm = landmarks_list[n]
            if lm:
                elbow = (int(lm[14].x * width), int(lm[14].y * height))
                wrist = (int(lm[16].x * width), int(lm[16].y * height))
                index = (int(lm[20].x * width), int(lm[20].y * height))
                hand = ((wrist[0] + index[0]) // 2, (wrist[1] + index[1]) // 2)
                arm_len = np.linalg.norm(np.array(elbow) - np.array(wrist))
            else:
                continue

            if arm_len < 20 or (prev_len and arm_len < prev_len * 0.2):
                release_fail_reason = f"팔 길이 조건 불충족: frame {n}, arm_len={arm_len}"
                continue

            if yolo_model is not None:
                results = yolo_model.predict(source=frame, conf=0.2, verbose=False)
                # YOLO 탐지 로그 추가
                ball_boxes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        ball_boxes.append([x1, y1, x2, y2])
                print(f"[YOLO] frame {n}: {len(ball_boxes)}개 탐지됨, 좌표: {ball_boxes}")
            else:
                results = []
            ball_pos, detected = None, False
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ball_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
                    detected = True
                    break

            if detected and not first_ball:
                first_ball = True
                prev_pos = ball_pos
            elif not detected:
                if not first_ball:
                    release_fail_reason = f"공 미탐지: frame {n}"
                    continue
                ball_pos = prev_pos

            if ball_pos is not None:
                dist = np.linalg.norm(np.array(hand) - np.array(ball_pos))
                if prev_dist and abs(dist - prev_dist) > 50:
                    release_fail_reason = f"손-공 거리 급변: frame {n}, dist={dist}, prev_dist={prev_dist}"
                    continue
                if prev_pos and np.linalg.norm(np.array(prev_pos) - np.array(ball_pos)) < 2:
                    release_fail_reason = f"공 위치 변화 미미: frame {n}"
                    continue
                if dist > arm_len * 0.5:
                    release_frame = n
                    break
                prev_dist, prev_pos, prev_len = dist, ball_pos, arm_len
    else:
        release_fail_reason = "fixed_frame 탐지 실패"

    if release_frame is None:
        print(f"[release 프레임 탐지 실패] reason: {release_fail_reason}")

    frame_list = [start_frame, max_knee_frame, fixed_frame, release_frame, (release_frame + 15) if release_frame is not None else None]

    return {
        'frame_list': frame_list,
        'fixed_frame': fixed_frame,
        'release_frame': release_frame,
        'frames': frames,
        'landmarks_list': landmarks_list,
        'width': width,
        'height': height
    }

# --- 2. 공 속도 및 궤적 시각화 ---
def visualize_ball_speed(video_path, release_frame, landmarks_list, yolo_model, width, height,
                         shin_length, save_path, SHIN_LENGTH_M=0.4, VIDEO_FPS=120):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)

    red_points = []
    first_pos = last_pos = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(source=frame, conf=0.5, verbose=False) if yolo_model else []
        ball_found = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                red_points.append((cx, cy))
                if first_pos is None:
                    first_pos = np.array([cx, cy])
                last_pos = np.array([cx, cy])
                ball_found = True
                break
        if not ball_found:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
    _, base_frame = cap.read()
    img = base_frame.copy()

    for pt in red_points:
        cv2.circle(img, pt, 5, (0, 0, 255), -1)

    if first_pos is not None and last_pos is not None and shin_length > 0:
        pixel_dist = np.linalg.norm(last_pos - first_pos)
        norm_dist = pixel_dist / shin_length
        frame_delta = len(red_points) - 1

        if frame_delta > 0:
            distance_m = norm_dist * SHIN_LENGTH_M
            time_s = frame_delta / VIDEO_FPS
            speed_kph = (distance_m / time_s) * 3.6

            text = f"Speed: {speed_kph:.1f} km/h (frames={frame_delta})"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
            cv2.rectangle(img, (20, 20), (20 + text_w + 20, 20 + text_h + 20), (255, 255, 255), -1)
            cv2.putText(img, text, (30, 20 + text_h + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)

    cv2.imwrite(save_path, img)
    return red_points
def get_ball_trajectory_and_speed(video_path, release_frame, yolo_model, width, height,
                                  shin_length, SHIN_LENGTH_M=0.4, VIDEO_FPS=120):
    import numpy as np
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)

    red_points = []
    first_pos = last_pos = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model.predict(source=frame, conf=0.5, verbose=False) if yolo_model else []
        ball_found = False
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                red_points.append((cx, cy))
                if first_pos is None:
                    first_pos = np.array([cx, cy])
                last_pos = np.array([cx, cy])
                ball_found = True
                break
        if not ball_found:
            break

    cap.release()

    speed_kph = None
    if first_pos is not None and last_pos is not None and shin_length > 0:
        pixel_dist = np.linalg.norm(last_pos - first_pos)
        norm_dist = pixel_dist / shin_length
        frame_delta = len(red_points) - 1
        if frame_delta > 0:
            distance_m = norm_dist * SHIN_LENGTH_M
            time_s = frame_delta / VIDEO_FPS
            speed_kph = (distance_m / time_s) * 3.6
    return {
        "trajectory": red_points,
        "speed_kph": speed_kph
    }

# --- 3. 각도/기울기 시각화 ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def get_joint_angles(lm, width, height):
    # 오른팔 (오른어깨-오른팔꿈치-오른손목)
    shoulder = (int(lm[12].x * width), int(lm[12].y * height))
    elbow = (int(lm[14].x * width), int(lm[14].y * height))
    wrist = (int(lm[16].x * width), int(lm[16].y * height))
    arm_angle = calculate_angle(shoulder, elbow, wrist)

    # 왼다리 (왼엉덩이-왼무릎-왼발목)
    left_hip = (int(lm[23].x * width), int(lm[23].y * height))
    left_knee = (int(lm[25].x * width), int(lm[25].y * height))
    left_ankle = (int(lm[27].x * width), int(lm[27].y * height))
    leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # 상체 기울기: 어깨중심 ~ 골반중심 벡터
    right_hip = (int(lm[24].x * width), int(lm[24].y * height))
    left_shoulder = (int(lm[11].x * width), int(lm[11].y * height))
    right_shoulder = (int(lm[12].x * width), int(lm[12].y * height))
    pelvis_center = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)
    shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
    vec = (pelvis_center[0] - shoulder_center[0], pelvis_center[1] - shoulder_center[1])
    tilt = math.degrees(math.atan2(vec[0], vec[1]))

    return {
        "arm_angle": arm_angle,
        "leg_angle": leg_angle,
        "tilt": tilt,
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "left_hip": left_hip,
        "left_knee": left_knee,
        "left_ankle": left_ankle,
        "pelvis_center": pelvis_center,
        "shoulder_center": shoulder_center
    }

def get_hand_height(lm, width, height, SHIN_LENGTH_M=0.4):
    ankle = (int(lm[27].x * width), int(lm[27].y * height))
    wrist = (int(lm[16].x * width), int(lm[16].y * height))
    knee = (int(lm[25].x * width), int(lm[25].y * height))
    shin_len = np.linalg.norm(np.array(knee) - np.array(ankle))
    dy = ankle[1] - wrist[1]
    normalized_height = dy / shin_len if shin_len > 0 else None
    real_height = normalized_height * SHIN_LENGTH_M if normalized_height is not None else None
    return {
        "normalized_height": normalized_height,
        "real_height": real_height,
        "wrist": wrist,
        "ankle": ankle,
        "knee": knee
    }

# --- 4. 평균폼 생성 및 유사도 분석 ---
def extract_and_normalize_landmarks(frames, landmarks_list, used_ids, width, height):
    normalized_landmarks = []
    for lm in landmarks_list:
        if lm is None:
            continue
        center_x = (lm[23].x + lm[24].x) / 2
        center_y = (lm[23].y + lm[24].y) / 2
        frame_coords = []
        for idx in used_ids:
            norm_x = lm[idx].x - center_x
            norm_y = lm[idx].y - center_y
            frame_coords.extend([norm_x, norm_y])
        normalized_landmarks.append(frame_coords)
    return np.array(normalized_landmarks)

def get_phase_ranges(frame_list):
    start, max_knee, fixed, release, follow = frame_list
    return [
        (start, max_knee),
        (max_knee, fixed),
        (fixed, release),
        (release, follow),
    ]

def generate_average_forms(video_paths, used_ids):
    phase_sequences = {i: [] for i in range(1, 5)}
    for path in video_paths:
        result = analyze_video(path)
        frame_list = result['frame_list']
        landmarks_list = result['landmarks_list']
        frames = result['frames']
        width = result['width']
        height = result['height']
        norm_seq = extract_and_normalize_landmarks(frames, landmarks_list, used_ids, width, height)
        phase_ranges = get_phase_ranges(frame_list)
        for i, (start_f, end_f) in enumerate(phase_ranges, 1):
            phase_seq = norm_seq[start_f:end_f]
            phase_sequences[i].append(phase_seq)
    avg_forms = {}
    for phase_idx, seq_list in phase_sequences.items():
        reference_seq = seq_list[0]
        aligned_seqs = []
        for seq in seq_list:
            path = fastdtw(reference_seq, seq, dist=euclidean)[1]
            aligned = [[] for _ in range(len(reference_seq))]
            for ref_idx, tgt_idx in path:
                aligned[ref_idx].append(seq[tgt_idx])
            avg_seq = [np.mean(frames, axis=0) if frames else np.zeros(reference_seq.shape[1]) for frames in aligned]
            aligned_seqs.append(np.array(avg_seq))
        avg_forms[phase_idx] = np.mean(np.stack(aligned_seqs), axis=0)
    return avg_forms

def score_from_distance(dist, min_dist=0.2, max_dist=5):
    if dist <= min_dist:
        return 100.0
    elif dist >= max_dist:
        return 0.0
    return 100.0 * (1 - (dist - min_dist) / (max_dist - min_dist))

def evaluate_against_average_form(test_video_path, average_forms, used_ids):
    result = analyze_video(test_video_path)
    frame_list = result['frame_list']
    landmarks_list = result['landmarks_list']
    frames = result['frames']
    width = result['width']
    height = result['height']
    norm_seq = extract_and_normalize_landmarks(frames, landmarks_list, used_ids, width, height)
    phase_ranges = get_phase_ranges(frame_list)
    phase_scores = []
    phase_distances = []
    for i, (start_f, end_f) in enumerate(phase_ranges, 1):
        test_phase_seq = norm_seq[start_f:end_f]
        avg_seq = average_forms[i]
        if len(test_phase_seq) == 0:
            dist = np.inf
        else:
            dist, _ = fastdtw(avg_seq, test_phase_seq, dist=euclidean)
        score = score_from_distance(dist)
        phase_scores.append(score)
        phase_distances.append(dist)
    worst_idx = np.argmin(phase_scores) + 1
    return phase_scores, phase_distances, worst_idx 