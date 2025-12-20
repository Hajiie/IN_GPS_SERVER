# IN-GPS Database Schema

## 1. Player (선수 정보)
선수의 기본 정보를 저장하는 테이블입니다.

| Field Name | Data Type | Constraints | Description |
|---|---|---|---|
| `id` | UUID | PK, Default: uuid4 | 선수 고유 식별자 |
| `number` | Integer | Nullable | 등번호 |
| `name` | CharField(100) | Not Null | 선수 이름 |
| `eng_name` | CharField(100) | Nullable | 선수 영문 이름 |
| `birth_date` | Date | Nullable | 생년월일 |
| `height` | Integer | Nullable | 키 (cm) |
| `weight` | Integer | Nullable | 몸무게 (kg) |
| `throwing_hand` | CharField(10) | Nullable, Choices: R/L | 투구 손 (우투/좌투) |
| `batting_hand` | CharField(10) | Nullable, Choices: R/L/S | 타격 손 (우타/좌타/양타) |
| `join_year` | Integer | Nullable | 입단 연도 |
| `career_stats` | CharField(100) | Nullable | 경력 사항 (예: XX초-OO중-AA고) |
| `playerImg` | ImageField | Nullable | 프로필 이미지 경로 |
| `playerStandImg` | ImageField | Nullable | 전신 이미지 경로 |
| `optimumForm` | UUID (FK) | Nullable, Ref: `VideoAnalysis.id` | 선수의 최적 폼 영상 ID |

## 2. PlayerSeason (선수 시즌 정보)
선수의 연도별 소속 팀 정보를 저장하는 테이블입니다.

| Field Name | Data Type | Constraints | Description |
|---|---|---|---|
| `id` | AutoField | PK | 시즌 정보 고유 ID |
| `player` | UUID (FK) | Not Null, Ref: `Player.id` | 선수 ID |
| `year` | Integer | Not Null | 시즌 연도 |
| `team` | CharField(50) | Not Null | 소속 팀명 |

* **Unique Constraint**: (`player`, `year`, `team`) 조합은 유일해야 합니다.

## 3. PlayerStats (선수 시즌 스탯)
특정 시즌의 투수 기록을 저장하는 테이블입니다. `PlayerSeason`과 1:1 관계를 가집니다.

| Field Name | Data Type | Constraints | Description |
|---|---|---|---|
| `id` | AutoField | PK | 스탯 정보 고유 ID |
| `player_season` | Integer (FK) | Not Null, Unique, Ref: `PlayerSeason.id` | 시즌 정보 ID |
| `era` | Decimal(4, 2) | Nullable | 평균자책점 (ERA) |
| `games` | Integer | Default: 0 | 경기 수 |
| `wins` | Integer | Default: 0 | 승리 |
| `losses` | Integer | Default: 0 | 패배 |
| `saves` | Integer | Default: 0 | 세이브 |
| `holds` | Integer | Default: 0 | 홀드 |
| `win_rate` | Decimal(5, 3) | Nullable | 승률 |
| `innings_pitched` | CharField(10) | Default: "0" | 이닝 수 |
| `hits_allowed` | Integer | Default: 0 | 피안타 |
| `home_runs_allowed` | Integer | Default: 0 | 피홈런 |
| `walks` | Integer | Default: 0 | 볼넷 |
| `hit_by_pitch` | Integer | Default: 0 | 사구 (몸에 맞는 공) |
| `strikeouts` | Integer | Default: 0 | 탈삼진 |
| `runs_allowed` | Integer | Default: 0 | 실점 |
| `earned_runs` | Integer | Default: 0 | 자책점 |

## 4. VideoAnalysis (영상 분석)
업로드된 투구 영상과 분석 결과를 저장하는 테이블입니다.

| Field Name | Data Type | Constraints | Description |
|---|---|---|---|
| `id` | UUID | PK, Default: uuid4 | 영상 고유 식별자 |
| `player` | UUID (FK) | Nullable, Ref: `Player.id` | 선수 ID |
| `video_name` | CharField(255) | Nullable, Unique | 영상 이름 |
| `video_file` | FileField | Not Null | 원본 영상 파일 경로 |
| `thumbnail` | ImageField | Nullable | 썸네일 이미지 경로 |
| `upload_time` | DateTime | Default: Now | 업로드 시간 |
| `fps` | Float | Nullable | 영상 FPS |
| `width` | Integer | Nullable | 영상 너비 |
| `height` | Integer | Nullable | 영상 높이 |
| `frame_list` | JSON | Nullable | 주요 프레임 인덱스 리스트 |
| `fixed_frame` | Integer | Nullable | 고정(Set) 프레임 인덱스 |
| `release_frame` | Integer | Nullable | 릴리스 포인트 프레임 인덱스 |
| `release_frame_knee` | JSON | Nullable | 릴리스 시 무릎 좌표 [x, y] |
| `release_frame_ankle` | JSON | Nullable | 릴리스 시 발목 좌표 [x, y] |
| `ball_speed` | JSON | Nullable | 구속 및 공 궤적 데이터 |
| `release_angle_height` | JSON | Nullable | 릴리스 각도 및 높이 데이터 |
| `skeleton_coords` | JSON | Nullable | 프레임별 스켈레톤 좌표 데이터 |
| `frame_metrics` | JSON | Nullable | 프레임별 신체 지표 (기울기, 각도 등) |
| `arm_trajectory` | JSON | Nullable | 팔 스윙 궤적 데이터 |
| `arm_swing_speed` | JSON | Nullable | 팔 스윙 속도 데이터 |
| `shoulder_swing_speed` | JSON | Nullable | 어깨 회전 속도 데이터 |
| `skeleton_video` | FileField | Nullable | 스켈레톤 오버레이 영상 경로 |
| `arm_swing_video` | FileField | Nullable | 팔 스윙 분석 영상 경로 |
| `shoulder_swing_video` | FileField | Nullable | 어깨 회전 분석 영상 경로 |
| `release_video` | FileField | Nullable | 릴리스 포인트 분석 영상 경로 |

## 5. DTWAnalysis (유사도 분석)
두 영상 간의 DTW(Dynamic Time Warping) 유사도 분석 결과를 저장하는 테이블입니다.

| Field Name | Data Type | Constraints | Description |
|---|---|---|---|
| `id` | UUID | PK, Default: uuid4 | 분석 결과 고유 ID |
| `player_id` | UUID (FK) | Nullable, Ref: `Player.id` | 선수 ID |
| `upload_time` | DateTime | Default: Now | 분석 시간 |
| `reference_video` | UUID | Not Null | 기준 영상(최적 폼) ID |
| `test_video` | UUID | Not Null | 테스트 영상 ID |
| `phase_scores` | JSON | Nullable | 구간별 유사도 점수 리스트 |
| `phase_distances` | JSON | Nullable | 구간별 거리값 리스트 |
| `overall_score` | Float | Nullable | 전체 유사도 점수 |
| `worst_phase` | Integer | Nullable | 가장 점수가 낮은 구간 인덱스 |
