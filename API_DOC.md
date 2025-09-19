# IN-GPS API 명세서 (v2.1)

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
    "id": "f0e9d8c7-b6a5-4321-fedc-ba9876543210",
    "name": "홍길동",
    "birth_date": "1995-03-15",
    "height": 185,
    "weight": 85,
    "throwing_hand": "R",
    "batting_hand": "R",
    "video_count": 3
  }
]
```

### **선수 등록**
`POST /analysis/players/create`
- **설명**: 새로운 선수를 등록합니다. `name`과 `birth_date`가 동일한 선수는 중복 등록할 수 없습니다.
- **요청 본문 (JSON)**:
```json
{
  "name": "이순신",
  "birth_date": "1998-01-10",
  "height": 180,
  "weight": 88,
  "throwing_hand": "L",
  "batting_hand": "L"
}
```
- **성공 응답**: `200 OK`
```json
{
  "result": "success",
  "player": {
    "id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "name": "이순신",
    "birth_date": "1998-01-10"
  }
}
```

### **개별 선수 조회**
`GET /analysis/players/<uuid:player_id>`
- **설명**: 특정 선수의 상세 정보를 조회합니다.
- **응답**: `200 OK`
```json
{
  "id": "f0e9d8c7-b6a5-4321-fedc-ba9876543210",
  "name": "홍길동",
  ...
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
    "reason": "이 선수와 연결된 영상이 3개 있습니다. 영상을 먼저 삭제해주세요."
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
    "team": "Toss-AI 히어로즈",
    "stats": {
      "era": "3.55", "games": 25, "wins": 10, ...
    }
  }
]
```

### **시즌 기록 생성/수정**
`POST /analysis/players/seasons/<uuid:player_id>`
- **설명**: 특정 선수의 시즌 기록과 스탯을 생성하거나 업데이트합니다. `year`와 `team`이 일치하는 기록이 있으면 스탯을 덮어쓰고, 없으면 새로 생성합니다.
- **요청 본문 (JSON)**:
```json
{
  "year": 2024,
  "team": "Toss-AI 히어로즈",
  "era": "3.55",
  "games": 25,
  "wins": 10
}
```
- **응답**: `200 OK`
```json
{
    "result": "success",
    "season_created": true,
    "stats_created": true,
    ...
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
  "video_url": "/media/videos/사용자-지정-영상-이름.mp4",
  "player_id": "f0e9d8c7-b6a5-4321-fedc-ba9876543210"
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
    "video_url": "/media/videos/사용자-지정-영상-이름.mp4",
    "upload_time": "2024-07-25T12:34:56Z",
    "player_id": "f0e9d8c7-b6a5-4321-fedc-ba9876543210",
    "player_name": "홍길동"
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
    "frame_list": [54, 122, 218, 228, 243],
    "fixed_frame": 218,
    "release_frame": 228,
    "ball_speed": { ... },
    "release_angle_height": { ... }
}
```

### **분석 결과 개별 조회**
- **설명**: `video_id`를 사용하여 특정 분석 결과를 조회합니다.
- **엔드포인트**:
  - `POST /analysis/ball_speed/<uuid:video_id>`
  - `POST /analysis/release_angle_height/<uuid:video_id>`
  - `POST /analysis/skeleton_coords/<uuid:video_id>`
- **응답**: `200 OK` (각 엔드포인트별 상세 분석 결과)

### **DTW 유사도 분석**
`POST /analysis/dtw_similarity`
- **설명**: 여러 영상을 기준으로 평균 폼을 생성하고 테스트 영상과 유사도를 비교합니다.
- **요청 본문 (JSON)**:
```json
{
  "average_ids": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5"],
  "test_id": "uuid6",
  "used_ids": [11, 12, 14, 16, 23, 24, 25] 
}
```
- **응답**: `200 OK`
```json
{
    "result": "success",
    "phase_scores": [82.8, 56.1, 4.8, 0.0],
    "phase_distances": [1.02, 2.30, 4.76, 8.81],
    "worst_phase": 4
}
```
