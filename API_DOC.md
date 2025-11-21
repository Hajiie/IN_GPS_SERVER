# IN-GPS API 명세서 (v2.2)

---

## 1. 개요

본 문서는 IN-GPS 투수 분석 시스템의 백엔드 API를 설명합니다. API는 선수 관리, 영상 관리, 시즌 기록 관리, 투구 분석 등 다양한 기능을 제공합니다. 모든 API의 기본 경로는 `/analysis/`로 시작합니다.

### 1.1. 주요 특징

- **보안**: 주요 리소스(선수, 영상)의 식별자로 추측 불가능한 **UUID**를 사용하여 IDOR 취약점을 방지합니다.
- **데이터 모델**: `선수(Player)`, `선수 시즌(PlayerSeason)`, `선수 스탯(PlayerStats)`, `영상 분석(VideoAnalysis)`의 4가지 주요 모델을 중심으로 구성됩니다.
- **분석 기능**: 업로드된 영상을 기반으로 투구 동작 분할, 공 속도 및 궤적, 릴리스 각도, 유사도(DTW) 등 다양한 분석을 수행하고 결과를 제공합니다.

---

## 2. 선수 (Player)

선수 개인의 기본 정보를 관리합니다.

### **선수 목록 조회**
`GET /analysis/players`
- **설명**: 등록된 모든 선수의 목록을 조회합니다.
- **응답**: `200 OK`
```json
[
    {
        "id": "aed08db3-13cd-4cc9-aebb-a91d090312ef",
        "number": "1",
        "name": "홍길동",
        "eng_name": "Hong Gil Dong",
        "playerImg": "/media/player_images/선수UUID/선수이미지UUID.png",
        "birth_date": "1995-03-15",
        "height": 185,
        "weight": 85,
        "throwing_hand": "R",
        "batting_hand": "R",
        "video_count": 0,
        "team_name": "TestTeam",
        "join_year": 2021,
        "career_stats": "XX초-OO중-AA고"
    },...
]
```

### **선수 등록**
`POST /analysis/players/create`
- **설명**: 새로운 선수를 등록합니다. `name`과 `birth_date`가 동일한 선수는 중복 등록할 수 없습니다.
- **요청 본문 (Multipart/Form-Data)**:
  - `pimage` (File): 프로필 이미지 파일
  - `simage` (File): 상세 이미지 파일
  - `name` (String): 선수 이름
  - `eng_name` (String): 선수 영문명
  - `number` (Int): 선수 등번호
  - `birth_date` (String): 선수 생년월일 (YYYY-MM-DD)
  - `height` (Int): 선수 키
  - `weight` (Int): 선수 체중
  - `throwing_hand` (String): 피칭 시 사용하는 손
  - `batting_hand` (String): 타격 시 사용하는 손
- **성공 응답**: `200 OK`
```json
{
    "result": "success",
    "player": {
        "id": "aed08db3-13cd-4cc9-aebb-a91d090312ef",
        "name": "홍길동",
        "birth_date": "1995-03-15"
    }
}
```

### **개별 선수 조회**
`GET /analysis/players/<uuid:player_id>`
- **설명**: 특정 선수의 상세 정보를 조회합니다.
- **응답**: `200 OK`
```json
{
    "id": "aed08db3-13cd-4cc9-aebb-a91d090312ef",
    "number": "1",
    "name": "홍길동",
    "eng_name": "Hong Gil Dong",
    "playerStandImg": "/media/player_standing_images/선수UUID/선수스탠딩이미지.png",
    "birth_date": "1995-03-15",
    "height": 185,
    "weight": 85,
    "throwing_hand": "R",
    "batting_hand": "R",
    "video_count": 0,
    "team_name": "팀명",
    "join_year": 2021,
    "career_stats": "XX초-OO중-AA고"
}
```

### **선수 정보 수정**
`PUT /analysis/players/update/<uuid:player_id>`
- **설명**: 특정 선수의 정보를 수정합니다. (수정할 필드만 포함하여 요청)
- **요청 본문 (JSON)**:
```json
{ "weight": 90 }
```
- **응답**: `200 OK` (수정된 선수의 전체 정보)
```json
{
    "result": "success",
    "player": {
        "id": "aed08db3-13cd-4cc9-aebb-a91d090312ef",
        "name": "홍길동",
        "birth_date": "1995-03-15",
        "height": 185,
        "weight": 90,
        "throwing_hand": "R",
        "batting_hand": "R"
    }
}
```

### **선수 삭제**
`DELETE /analysis/players/delete/<uuid:player_id>`
- **설명**: 특정 선수를 삭제합니다. 해당 선수에게 연결된 영상이 하나라도 있으면 삭제할 수 없습니다.
- **성공 응답**: `200 OK`
```json
{ "result": "success" }
```
- **실패 응답 (연결된 영상 존재 시)**: `400 Bad Request`
```json
{
    "result": "fail",
    "reason": "이 선수와 연결된 영상이 1개 있습니다. 영상을 먼저 삭제해주세요."
}
```

### **최적폼 설정**
`POST /analysis/players/optimum_form/<uuid:player_id>`
- **설명**: 특정 선수의 최적폼을 설정합니다.
- **요청 본문 (JSON)**:

```json
{
  "video_id": "d1e2f3a4-b5c6-7890-1234-567890abcdef"
}
```
- **응답**: `200 ok`
```json
{
    "result": "success"
}
```

---

## 3. 선수 시즌 기록 (Player Season)

선수별 시즌 기록과 상세 스탯을 관리합니다.

### **시즌 기록 조회**
`GET /analysis/players/seasons/<uuid:player_id>`
- **설명**: 특정 선수의 모든 시즌 기록과 스탯을 연도 내림차순으로 조회합니다.
- **응답**: `200 OK`
```json
[
    {
        "season_id": 1,
        "year": 2024,
        "team": "TestTeam",
        "stats": {
            "era": "3.55",
            "games": 25,
            "wins": 10,
            "losses": 5,
            "saves": 2,
            "holds": 0,
            "win_rate": "0.667",
            "innings_pitched": "0.0",
            "hits_allowed": 0,
            "home_runs_allowed": 0,
            "walks": 0,
            "hit_by_pitch": 0,
            "strikeouts": 150,
            "runs_allowed": 0,
            "earned_runs": 0
        }
    },...
]
```

### **시즌 기록 생성/수정**
`POST /analysis/players/seasons/<uuid:player_id>`
- **설명**: 특정 선수의 시즌 기록과 스탯을 생성하거나 업데이트합니다. `year`와 `team`이 일치하는 기록이 있으면 스탯을 덮어쓰고, 없으면 새로 생성합니다.
- **요청 본문 (JSON)**:
```json
{
    "year": 2024,
    "team": "TestTeam",
    "era": "3.55",
    "games": 25,
    "wins": 10,
    "losses": 5,
    "saves": 2,
    "strikeouts": 150
}
```
- **응답**: `200 OK`
```json
{
    "result": "success",
    "season_created": true,
    "stats_created": true,
    "player_id": "8fab596a-ae0b-4342-b974-180e3f498029",
    "season_id": 1,
    "year": 2024,
    "team": "TestTeam",
    "stats": {
        "era": "3.55",
        "games": 25,
        "wins": 10,
        "losses": 5,
        "saves": 2,
        "holds": 0,
        "win_rate": 0.6666666666666666,
        "innings_pitched": "0.0",
        "hits_allowed": 0,
        "home_runs_allowed": 0,
        "walks": 0,
        "hit_by_pitch": 0,
        "strikeouts": 150,
        "runs_allowed": 0,
        "earned_runs": 0
    }
}
```

---

## 4. 영상 (Video)

투구 영상의 업로드, 조회, 삭제를 관리합니다.

### **영상 업로드**
`POST /analysis/upload`
- **요청 본문 (Multipart/Form-Data)**:
  - `video` (File): 영상 파일
  - `player_name` (String): 선수 이름
  - `birth_date` (String): 선수 생년월일 (YYYY-MM-DD)
  - `video_name` (String, Optional): 사용자 지정 영상 이름. **(고유해야 함)**
- **성공 응답**: `200 OK`
```json
{
  "result": "success",
  "video_id": "d1e2f3a4-b5c6-7890-1234-567890abcdef",
  "video_name": "사용자 지정 영상 이름",
  "video_url": "/media/videos/<uuid:player_id>/<uuid:video_id>.mp4",
  "thumbnail_url": "/media/thumbnails/<uuid:player_id>/<uuid:thumbnail_id>.jpg",
  "player_id": "f0e9d8c7-b6a5-4321-fedc-ba9876543210",
  "player_name": "선수이름"
}
```
- **실패 응답 (이름 중복 시)**: `409 Conflict`
```json
{
    "result": "fail",
    "reason": "이미 사용 중인 영상 이름입니다: \"중복된 이름\""
}
```

### **영상 목록 조회**
`GET /analysis/videos`
- **설명**: 업로드된 모든 영상 목록을 최신순으로 조회합니다.
- **응답**: `200 OK`
```json
[
  {
    "id": "d1e2f3a4-b5c6-7890-1234-567890abcdef",
    "video_name": "사용자 지정 영상 이름",
    "video_url": "/media/videos/<uuid:player_id>/<uuid:video_id>.mp4",
    "thumbnail_url": "/media/thumbnails/<uuid:player_id>/<uuid:thumbnail_id>.jpg",
    "upload_time": "2024-07-25T12:34:56Z",
    "player_id": "f0e9d8c7-b6a5-4321-fedc-ba9876543210",
    "player_name": "홍길동",
    "isAnalysis": true
  }
]
```

### **영상 삭제**
`DELETE /analysis/videos/delete/<uuid:video_id>`
- **설명**: 특정 영상을 서버와 데이터베이스에서 삭제합니다.
- **응답**: `200 OK`
```json
{
  "result": "success",
  "message": "Video \"사용자 지정 영상 이름\" deleted successfully."
}
```

### **영상 상세 조회**
`GET /analysis/videos/<uuid:video_id>`
- **설명**: 특정 영상의 상세 정보를 조회합니다.
- **응답**: `200 OK`
```json
{
    "result": "success",
    "video_name": "test03_01",
    "video_url": "/media/videos/<uuid:player_id>/<uuid:video_id>.mp4",
    "thumbnail_url": "/media/thumbnails/<uuid:player_id>/<uuid:thumbnail_id>.jpg",
    "upload_time": "2025-10-28T14:32:47.635134+00:00",
    "player_id": "89688ec3-ae73-4598-9bb7-d6edc98d4bd7",
    "player_name": "홍길동",
    "ball_speed": {
        "trajectory": [[1261, 320],...],
        "speed_kph": 92.14018923830845
    },
    "release_frame": 226,
    "width": 1920,
    "height": 1080,
    "release_frame_knee": [1095, 768],
    "release_frame_ankle": [1143, 894],
    "fixed_frame": 213,
    "frame_list": [32, 99, 213, 226, 241],
    "skeleton_video_url": "/media/skeleton_videos/<uuid:player_id>/<uuid:skeleton_id>.mp4",
    "arm_video_url": "/media/arm_swing_videos/<uuid:player_id>/<uuid:arm_id>.mp4",
    "shoulder_video_url": "/media/shoulder_swing_videos/<uuid:player_id>/<uuid:shoulder_id>.mp4",
    "release_video_url": "/media/release_videos/<uuid:player_id>/<uuid:release_id>.mp4",
    "release_angle_height": {
        "angles": {
            "arm_angle": 150.49521009982092,
            "leg_angle": 143.0501919742916,
            "tilt": -23.080617215343793,
            "shoulder": [1107, 514],
            "elbow": [1192, 468],
            "wrist": [1239, 393],
            "left_hip": [1014, 717],
            "left_knee": [1095, 768],
            "left_ankle": [1143, 894],
            "pelvis_center": [1010, 709],
            "shoulder_center": [1085, 533]
        },
        "hand_height": {
            "normalized_height": 3.715701231741373,
            "real_height": 1.6720655542836178,
            "wrist": [1239, 393],
            "ankle": [1143, 894],
            "knee": [1095, 768]
        }
    },
    "frame_metrics": [
        {
            "torso_tilt": -0.6127657222640064,
            "elbow_angle": 30.809689973132183,
            "knee_angle": 177.70802307090733,
            "hand_height_m": 1.426872335668805
        },...
    ],
    "arm_trajectory": [[952, 416], ...],
    "arm_swing_speed": [1.742704666616396, ...],
    "shoulder_swing_speed": [-461.0091283506458, ...],
    "fps": 30
}
```

---

## 5. 분석 (Analysis)

업로드된 영상을 분석하고 결과를 조회합니다.

### **영상 분석 실행**
`POST /analysis/analyze/<uuid:video_id>`
- **설명**: 특정 영상에 대한 전체 분석(동작 분할, 공 속도, 릴리스 각도 등)을 실행하고 결과를 DB에 저장합니다.
- **응답**: `200 OK` (분석 결과 요약)
```json
{
    "result": "success",
    "frame_list": [51, 111, 220, 236, 251],
    "fixed_frame": 220,
    "release_frame": 236,
    "ball_speed": {
        "trajectory": [
            [
                1282,
                339
            ],...
        ],
        "speed_kph": 96.48510214059976
    },
    "release_angle_height": {
        "angles": {
            "arm_angle": 142.58744340070731,
            "leg_angle": 125.38037670563261,
            "tilt": -22.34810834790941,
            "shoulder": [1118, 518],
            "elbow": [1207, 486],
            "wrist": [1256, 410],
            "left_hip": [1021, 718],
            "left_knee": [1114, 762],
            "left_ankle": [1136, 886],
            "pelvis_center": [1021, 716],
            "shoulder_center": [1095, 536]
        },
        "hand_height": {
            "normalized_height": 3.77968285998549,
            "real_height": 1.7008572869934706,
            "wrist": [1256, 410],
            "ankle": [1136, 886],
            "knee": [1114, 762]
        }
    }
}
```

### **분석 결과 개별 조회**
- **설명**: `video_id`를 사용하여 특정 분석 결과를 조회합니다.
- **엔드포인트**:
  - `GET /analysis/ball_speed/<uuid:video_id>`
  - `GET /analysis/release_angle_height/<uuid:video_id>`
  - `GET /analysis/skeleton_coords/<uuid:video_id>`
  - `GET /analysis/frame_metrics/<uuid:video_id>`
  - `GET /analysis/arm_trajectory/<uuid:video_id>`
- **응답**: `200 OK` (각 엔드포인트별 상세 분석 결과)
```json
{
  "result": "success",
  "release_angle_height": {
        "angles": {
            "arm_angle": 142.58744340070731,
            "leg_angle": 125.38037670563261,
            "tilt": -22.34810834790941,
            "shoulder": [1118, 518],
            "elbow": [1207, 486],
            "wrist": [1256, 410],
            "left_hip": [1021, 718],
            "left_knee": [1114, 762],
            "left_ankle": [1136, 886],
            "pelvis_center": [1021, 716],
            "shoulder_center": [1095, 536]
        },
        "hand_height": {
            "normalized_height": 3.77968285998549,
            "real_height": 1.7008572869934706,
            "wrist": [1256, 410],
            "ankle": [1136, 886],
            "knee": [1114, 762]
        }
    }
}
```
공을 추적하는 순간 프레임부터 끝까지, 공의 속도 및 각 프레임에 대해 공의 x_pixel,y_pixel의 위치를 반환
```json
{
  "result": "success",
  "ball_speed": {
        "trajectory": [
            [
                1282,
                339
            ],...
        ],
        "speed_kph": 96.48510214059976
    }
}
```
모든 프레임에 대한 33개의 랜드마크와 각 랜드마크별 `x`,`y`,`visibility`,`x_pixel`,`y_pixel`을 반환
```json
{
    "result": "success",
    "skeleton_coords": [
        [
            {
                "x": 0.42583516240119934,
                "y": 0.3370037376880646,
                "visibility": 0.9986435770988464,
                "x_pixel": 817,
                "y_pixel": 363
            },...
        ],...
    ]
}
```
모든 프레임에 대한 몸 기울기, 팔꿈치 각도, 무릎 각도, 손 높이를 반환
```json
{
    "result": "success",
    "frame_metrics": [
        {
            "torso_tilt": -0.6127657222640064,
            "elbow_angle": 30.809689973132183,
            "knee_angle": 177.70802307090733,
            "hand_height_m": 1.426872335668805
        },...
    ]
}
```
2구간에서 4구간까지의 팔 스윙 궤적, 팔 스윙 속도, 어깨 회전 속도를 반환
```json
{
    "result": "success",
    "arm_trajectory": [[913,425],...],
    "arm_swing_speed": [1.3865232754997097,...],
    "shoulder_swing_speed": [-310.38960957989036,...]
}
```

### **DTW 유사도 분석**
`POST /analysis/dtw_similarity`
- **설명**: 최적폼 영상을 기준으로 테스트 영상과 유사도를 비교합니다. `used_ids`는 비워도 되는 필드입니다.
- **요청 본문 (JSON)**: 
```json
{
    "reference_id": "3f817408-87a9-46d6-8051-82bde67b898d",
    "test_id": "5b3d9135-9df1-4951-8a88-eaf543288767",
    "used_ids": [11, 12, 14, 16, 23, 24, 25] 
}
```
- **응답**: `200 OK`
```json
{
    "result": "success",
    "phase_scores": [87.94750010042343, 88.02798111679282, 86.16201109493117, 74.41059810364298],
    "phase_distances": [
      0.018078749849364858, 0.01795802832481077, 
      0.020756983357603228, 0.038384102844535535
    ],
    "overall_score": 84.1370226039476,
    "worst_phase": 4
}
```

### **DTW 유사도 분석 점수 기록 반환**
`GET /analysis/dtw_similarity/<uuid:player_id>`
- **설명**: 훈련점수 기록
- **응답**: `200 OK`
```json
{
    "result": "success",
    "dtw_similarity_scores": [
        {
            "upload_time": "2025-11-21",
            "overall_score": 100.0,
            "isChanged": false
        }, ...
    ]
}
```