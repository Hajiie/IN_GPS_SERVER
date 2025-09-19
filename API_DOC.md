# IN-GPS API 명세서

**Version: 2.0**

- **주요 변경사항 (v2.0)**
  - 모든 API의 ID를 추측 불가능한 고유 식별자 **UUID**로 변경하여 보안을 강화했습니다.
  - 영상 삭제 API가 추가되었습니다.
  - 선수별 시즌 기록을 생성/수정/조회하는 API가 추가되었습니다.
  - 영상 업로드 시 사용자 지정 이름을 사용할 수 있도록 개선되었습니다.

---

## 1. 선수 (Player)

### **선수 목록 조회**
`GET /api/players/`
- **설명**: 등록된 모든 선수의 목록을 조회합니다.
- **응답**:
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

### **개별 선수 조회**
`GET /api/players/<uuid:player_id>/`
- **설명**: 특정 선수의 상세 정보를 조회합니다.
- **응답**:
```json
{
  "id": "f0e9d8c7-b6a5-4321-fedc-ba9876543210",
  "name": "홍길동",
  "birth_date": "1995-03-15",
  ...
}
```

### **선수 등록**
`POST /api/players/create/`
- **설명**: 새로운 선수를 등록합니다.
- **요청 본문 (JSON)**:
```json
{
  "name": "이순신",
  "birth_date": "1998-01-10",
  "height": 180,
  "weight": 88
}
```
- **응답**:
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

### **선수 정보 수정**
`PUT /api/players/<uuid:player_id>/update/`
- **설명**: 특정 선수의 정보를 수정합니다. (수정할 필드만 포함)
- **요청 본문 (JSON)**:
```json
{
  "height": 181,
  "weight": 90
}
```
- **응답**: 수정된 선수의 전체 정보

### **선수 삭제**
`DELETE /api/players/<uuid:player_id>/delete/`
- **설명**: 특정 선수를 삭제합니다. (연결된 영상이 없어야 함)
- **응답**:
```json
{ "result": "success" }
```

---

## 2. 선수 시즌 기록 (Player Season)

### **시즌 기록 조회**
`GET /api/players/<uuid:player_id>/seasons/`
- **설명**: 특정 선수의 모든 시즌 기록과 스탯을 조회합니다.
- **응답**:
```json
[
  {
    "season_id": 1,
    "year": 2024,
    "team": "Toss-AI 히어로즈",
    "stats": {
      "era": "3.55",
      "games": 25,
      "wins": 10,
      ...
    }
  }
]
```

### **시즌 기록 생성/수정**
`POST /api/players/<uuid:player_id>/seasons/`
- **설명**: 특정 선수의 시즌 기록과 스탯을 생성하거나 업데이트합니다.
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
- **응답**: 처리 결과 및 생성/수정된 시즌 정보

---

## 3. 영상 (Video)

### **영상 업로드**
`POST /api/upload/`
- **설명**: 비디오 파일을 업로드하고 분석 대기열에 추가합니다.
- **요청 본문 (Multipart/Form-Data)**:
  - `video` (File): 영상 파일
  - `player_name` (String): 선수 이름
  - `birth_date` (String): 선수 생년월일 (YYYY-MM-DD)
  - `video_name` (String, Optional): 사용자 지정 영상 이름
- **응답**:
```json
{
  "result": "success",
  "video_id": "d1e2f3a4-b5c6-7890-1234-567890abcdef",
  "video_name": "사용자 지정 영상 이름",
  "video_url": "/media/videos/사용자-지정-영상-이름.mp4",
  "player_id": "f0e9d8c7-b6a5-4321-fedc-ba9876543210",
  "player_name": "홍길동"
}
```

### **영상 목록 조회**
`GET /api/videos/`
- **설명**: 업로드된 모든 영상 목록을 조회합니다.
- **응답**:
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
`DELETE /api/videos/<uuid:video_id>/delete/`
- **설명**: 특정 영상을 서버와 데이터베이스에서 삭제합니다.
- **응답**:
```json
{
  "result": "success",
  "message": "Video \"사용자 지정 영상 이름\" deleted successfully."
}
```

---

## 4. 분석 (Analysis)

### **영상 분석 실행**
`POST /api/analyze/`
- **설명**: 업로드된 영상에 대한 전체 분석을 실행하고 결과를 DB에 저장합니다.
- **요청 본문 (JSON)**:
```json
{
  "video_id": "d1e2f3a4-b5c6-7890-1234-567890abcdef"
}
```
- **응답**: 분석 결과 요약 (상세 결과는 개별 API로 조회)

### **분석 결과 개별 조회**
- **설명**: `video_id`를 사용하여 특정 분석 결과를 조회합니다.
- **엔드포인트 및 요청/응답 형식은 이전 버전과 유사하나, 요청 시 Body에 포함되는 `id`는 이제 **UUID** 형식이어야 합니다.**
  - `POST /api/ball_speed/`
  - `POST /api/release_angle_height/`
  - `POST /api/skeleton_coords/`
- **요청 예시 (`/api/ball_speed/`)**:
```json
{
  "id": "d1e2f3a4-b5c6-7890-1234-567890abcdef"
}
```

### **DTW 유사도 분석**
`POST /api/dtw_similarity/`
- **설명**: 여러 영상을 기준으로 평균 폼을 생성하고 테스트 영상과 유사도를 비교합니다.
- **요청 본문 (JSON)**: `average_ids`와 `test_id`에 영상의 UUID를 사용합니다.
```json
{
  "average_ids": ["uuid1", "uuid2", "uuid3"],
  "test_id": "uuid4"
}
```
- **응답**: 이전 버전과 동일
