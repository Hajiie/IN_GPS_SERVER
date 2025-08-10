# 투수 분석 백엔드 API 명세서

## 1. 기능 요약

- **Django 기반 백엔드 구축**
  - 분석용 앱(`analysis`) 구성 및 URL 라우팅

- **선수(Player) 테이블 및 선수별 영상 관리**
  - 선수 정보를 별도 테이블(`Player`)로 관리
  - 영상(분석) 테이블(`VideoAnalysis`)에서 선수(`player`)를 외래키로 참조
  - 업로드/조회/분석 시 선수 지정 및 선수별 영상 관리 가능

- **영상 업로드 및 저장**
  - 업로드된 영상을 서버(`media/videos/`)에 저장
  - 업로드 시 DB(`VideoAnalysis`)에 row가 자동 생성됨 (선수 지정 가능)

- **기본 분석 및 결과 저장**
  - `/analysis/analyze/` 호출 시 모든 분석 결과(ball_speed, release_angle_height, skeleton_coords 등)가 DB에 자동 저장됨

- **분석 결과 조회**
  - ball_speed, release_angle_height, skeleton_coords 등은 영상 id(pk)로 조회

- **공 궤적 및 속도 분석**
  - **YOLO 모델(`model_path/best_baseball_ball.pt`)을 사용**하여 공 궤적 추적, 속도 계산
  - **이미지 저장 없이, 궤적 좌표와 속도를 JSON으로 반환**

- **릴리스 프레임 각도/기울기/손 높이 분석**
  - 릴리스 프레임에서 각도, 기울기, 손 높이 등 **수치 데이터(좌표, 각도, 길이 등)를 JSON으로 반환**

- **평균폼 생성 및 유사도(DTW) 분석**
  - 좋은 폼 영상 5개로 평균폼 생성
  - 테스트 영상과의 DTW 기반 단계별 유사도 점수/거리/최저점 단계 반환

- **스켈레톤 좌표 반환**
  - **영상 전체 프레임별** 모든 랜드마크(관절) 좌표를 JSON으로 반환

---

## 2. API 명세서

### 0) 선수 관리

#### 선수 목록 조회
**GET** `/analysis/players/`
- **응답**
```json
[
  {
    "id": 1,
    "name": "홍길동",
    "birth_date": "1995-03-15",
    "height": 185,
    "weight": 85,
    "throwing_hand": "R",
    "batting_hand": "R",
    "video_count": 3
  },
  {
    "id": 2,
    "name": "김철수",
    "birth_date": "1998-07-22",
    "height": 180,
    "weight": 80,
    "throwing_hand": "L",
    "batting_hand": "L",
    "video_count": 1
  }
]
```

#### 개별 선수 조회
**GET** `/analysis/players/{player_id}/`
- **응답**
```json
{
  "id": 1,
  "name": "홍길동",
  "birth_date": "1995-03-15",
  "height": 185,
  "weight": 85,
  "throwing_hand": "R",
  "batting_hand": "R",
  "video_count": 3
}
```

#### 선수 등록
**POST** `/analysis/players/create/`
- **Body (JSON)**
```json
{
  "name": "홍길동",
  "birth_date": "1995-03-15",
  "height": 185,
  "weight": 85,
  "throwing_hand": "R",
  "batting_hand": "R"
}
```
- **응답**
```json
{
  "result": "success",
  "player": {
    "id": 1,
    "name": "홍길동",
    "birth_date": "1995-03-15",
    "height": 185,
    "weight": 85,
    "throwing_hand": "R",
    "batting_hand": "R"
  }
}
```

#### 선수 정보 수정
**PUT** `/analysis/players/{player_id}/update/`
- **Body (JSON)** - 수정할 필드만 포함
```json
{
  "height": 186,
  "weight": 87
}
```
- **응답**
```json
{
  "result": "success",
  "player": {
    "id": 1,
    "name": "홍길동",
    "birth_date": "1995-03-15",
    "height": 186,
    "weight": 87,
    "throwing_hand": "R",
    "batting_hand": "R"
  }
}
```

#### 선수 삭제
**DELETE** `/analysis/players/{player_id}/delete/`
- **응답**
```json
{
  "result": "success"
}
```
- **에러 응답** (연결된 영상이 있는 경우)
```json
{
  "result": "fail",
  "reason": "이 선수와 연결된 영상이 3개 있습니다. 영상을 먼저 삭제해주세요."
}
```

---

### 1) 영상 업로드  
**POST** `/analysis/upload/`
- **폼 필드명**: 
  - `video` (파일): 업로드할 영상 파일
  - `player_name` (필수): 선수 이름
  - `birth_date` (필수): 선수 생년월일 (YYYY-MM-DD 형식)
- **동작**: 
  - 선수 이름과 생년월일로 기존 선수를 찾거나 새로 생성
  - 업로드 시 DB에 영상 row가 자동 생성됨 (선수 정보 포함)
- **응답**
```json
{
  "result": "success",
  "filename": "업로드된파일명.mp4",
  "path": "저장경로",
  "player_id": 1,
  "player_name": "홍길동",
  "player_created": false
}
```
- **에러 응답**
```json
{
  "result": "fail",
  "reason": "선수 이름과 생년월일을 모두 입력해주세요"
}
```

---

### 1-1) 업로드된 영상 목록 조회 (선수별 필터 가능)
**GET** `/analysis/videos/`  
- **쿼리 파라미터**: `player_id` (선택)
- **응답**
```json
[
  { "id": 1, "filename": "20250721_test.mp4", "upload_time": "2025-07-21T12:34:56Z", "player": 1 },
  ...
]
```

---

### 2) 기본 분석 및 결과 저장  
**POST** `/analysis/analyze/`
- **Body (JSON)**
```json
{ "filename": "업로드된파일명.mp4" }
```
- **동작**: 분석 결과(ball_speed, release_angle_height, skeleton_coords 등)가 DB에 자동 저장됨
- **응답 예시**
```json
{
    "result": "success", // 분석 성공 여부
    "frame_list": [54, 122, 218, 228, 243], // 투구 동작 주요 구간 프레임 인덱스
    "fixed_frame": 218, // 디딤발 고정 프레임
    "release_frame": 228, // 릴리스 프레임
    "width": 1920, // 영상 가로 해상도
    "height": 1080, // 영상 세로 해상도
    "release_frame_knee": [1139, 783], // 릴리스 프레임에서 무릎 좌표 (픽셀)
    "release_frame_ankle": [1171, 895], // 릴리스 프레임에서 발목 좌표 (픽셀)
    "ball_speed": {
        "trajectory": [[1271, 319], ...], // 공 궤적 좌표 리스트 (픽셀)
        "speed_kph": 85.59 // 공 속도 (km/h)
    },
    "release_angle_height": {
        "angles": {
            "arm_angle": 149.25, // 오른팔 각도 (도)
            "leg_angle": 140.32, // 왼다리 각도 (도)
            "tilt": -21.28, // 상체 기울기 (도)
            "shoulder": [1145, 522], // 오른어깨 좌표 (픽셀)
            "elbow": [1225, 469], // 오른팔꿈치 좌표 (픽셀)
            "wrist": [1265, 386], // 오른손목 좌표 (픽셀)
            "left_hip": [1044, 718], // 왼엉덩이 좌표 (픽셀)
            "left_knee": [1139, 783], // 왼무릎 좌표 (픽셀)
            "left_ankle": [1171, 895], // 왼발목 좌표 (픽셀)
            "pelvis_center": [1045, 713], // 골반 중심 좌표 (픽셀)
            "shoulder_center": [1112, 541] // 어깨 중심 좌표 (픽셀)
        },
        "hand_height": {
            "normalized_height": 4.37, // 손 높이(정규화)
            "real_height": 1.74, // 손 높이(실제 m)
            "wrist": [1265, 386], // 손목 좌표 (픽셀)
            "ankle": [1171, 895], // 발목 좌표 (픽셀)
            "knee": [1139, 783] // 무릎 좌표 (픽셀)
        }
    },
    "skeleton_coords": [
      [// 0번 프레임
        {
            "x": 0.6019, // 랜드마크 x좌표 (정규화)
            "y": 0.4622, // 랜드마크 y좌표 (정규화)
            "visibility": 0.9996, // 관절 신뢰도(0~1)
            "x_pixel": 1155, // x좌표(픽셀)
            "y_pixel": 499 // y좌표(픽셀)
        },
    // ... 33개 랜드마크 좌표,
        ],
        [ // 1번 프레임
            {"x": 0.6020, "y": 0.4623, "visibility": 0.9995, "x_pixel": 1156, "y_pixel": 500},
            ...
        ],
        ...
    ],
    "player": 1, // 선수 id (nullable)
}
```

---

### 3) 공 궤적 및 속도 분석 결과 조회  
**POST** `/analysis/ball_speed/`
- **Body (JSON)**
```json
{ "id": 1 }
```
- **동작**: DB에서 해당 영상 id의 ball_speed 결과를 반환
- **응답 예시**
```json
{
    "result": "success", // 조회 성공 여부
    "ball_speed": {
        "speed_kph": 85.59, // 공 속도 (km/h)
        "trajectory": [[1271, 319], ...] // 공 궤적 좌표 리스트 (픽셀)
    }
}
```

---

### 4) 릴리스 각도/손 높이 등 수치 데이터 결과 조회  
**POST** `/analysis/release_angle_height/`
- **Body (JSON)**
```json
{ "id": 1 }
```
- **동작**: DB에서 해당 영상 id의 release_angle_height 결과를 반환
- **응답 예시**
```json
{
    "result": "success", // 조회 성공 여부
    "release_angle_height": {
        "angles": {
            "tilt": -21.28, // 상체 기울기 (도)
            "elbow": [1225, 469], // 팔꿈치 좌표 (픽셀)
            "wrist": [1265, 386], // 손목 좌표 (픽셀)
            "left_hip": [1044, 718], // 왼엉덩이 좌표 (픽셀)
            "shoulder": [1145, 522], // 어깨 좌표 (픽셀)
            "arm_angle": 149.25, // 팔 각도 (도)
            "left_knee": [1139, 783], // 왼무릎 좌표 (픽셀)
            "leg_angle": 140.32, // 다리 각도 (도)
            "left_ankle": [1171, 895], // 왼발목 좌표 (픽셀)
            "pelvis_center": [1045, 713], // 골반 중심 좌표 (픽셀)
            "shoulder_center": [1112, 541] // 어깨 중심 좌표 (픽셀)
        },
        "hand_height": {
            "knee": [1139, 783], // 무릎 좌표 (픽셀)
            "ankle": [1171, 895], // 발목 좌표 (픽셀)
            "wrist": [1265, 386], // 손목 좌표 (픽셀)
            "real_height": 1.74, // 손 높이(실제 m)
            "normalized_height": 4.37 // 손 높이(정규화)
        }
    }
}
```

---

### 5) 영상 전체 프레임별 스켈레톤(랜드마크) 좌표 결과 조회  
**POST** `/analysis/skeleton_coords/`
- **Body (JSON)**
```json
{ "id": 1 }
```
- **동작**: DB에서 해당 영상 id의 skeleton_coords 결과(전체 프레임별 랜드마크 좌표 리스트)를 반환
- **응답 예시**
```json
{
    "result": "success", // 조회 성공 여부
    "skeleton_coords": [
        [ // 0번 프레임
            {"x": 0.6019, "y": 0.4622, "visibility": 0.9996, "x_pixel": 1155, "y_pixel": 499},
            ... // 33개 랜드마크
        ],
        [ // 1번 프레임
            {...}, ...
        ],
        ...
    ]
}
```
- **설명**: 각 프레임별로 33개 랜드마크의 좌표 리스트가 들어있음. None이면 해당 프레임에서 랜드마크 추출 실패.

---

### 6) 평균폼 생성 및 유사도(DTW) 분석  
**POST** `/analysis/dtw_similarity/`
- **Body (JSON)**
```json
{
  "average_ids": [1, 2, 3, 4, 5],
  "test_id": 6,
  "used_ids": [11, 12, 14, 16, 23, 24, 25] // (생략 가능)
}
```
- **응답 예시**
```json
{
    "result": "success",
    "phase_scores": [
        82.89336241395954,
        56.128093259270194,
        4.872353732228296,
        0.0
    ],
    "phase_distances": [
        1.021118604129942,
        2.3058515235550305,
        4.766127020853042,
        8.810676068085149
    ],
    "worst_phase": 4
}
```
- **설명**
  - average_ids: 평균폼을 만들 기준이 되는 영상 id 리스트
  - test_id: 비교할 영상 id
  - used_ids: 분석에 사용할 관절 id 리스트(생략 시 기본값 사용)
  - phase_scores: 각 구간별 유사도 점수(0~100, 높을수록 유사)
  - phase_distances: 각 구간별 DTW 거리(낮을수록 유사)
  - worst_phase: 가장 유사도가 낮은 구간(1~4) 