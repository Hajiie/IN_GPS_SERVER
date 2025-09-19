# IN-GPS Pitching Analysis Backend

**IN-GPS**는 Django 기반의 투수 동작 분석 백엔드 서버입니다. 영상 업로드, 선수 관리, 투구 분석 등 다양한 기능을 RESTful API로 제공하여, 클라이언트 애플리케이션에서 투수의 데이터를 체계적으로 관리하고 분석 결과를 활용할 수 있도록 지원합니다.

---

##  주요 기능

- **선수 관리**: 선수 정보(이름, 신체 정보, 투타 방향 등)의 생성, 조회, 수정, 삭제 (CRUD)
- **시즌 기록 관리**: 선수별 시즌 성적 및 상세 스탯 관리
- **영상 관리**: 투구 영상 업로드, 조회, 삭제 및 영상에 대한 메타데이터 관리
- **투구 분석**: MediaPipe와 YOLO를 활용한 자동 분석
  - **투구 동작 분할**: 투구의 주요 4단계(Start, Max Knee, Fixed, Release) 프레임 자동 검출
  - **공 속도 및 궤적**: 릴리스 이후의 공의 궤적을 추적하고 속도(km/h) 계산
  - **릴리스 포인트 분석**: 릴리스 순간의 팔 각도, 상체 기울기, 손 높이 등 상세 데이터 추출
  - **유사도 분석 (DTW)**: 기준 영상들과의 투구폼 유사도를 구간별로 점수화
- **간편 실행**: `run_server.py`를 통해 데이터베이스 자동 생성 및 서버 실행을 한 번에 처리

---

## 시작하기

이 섹션은 프로젝트를 로컬 환경에서 설정하고 실행하는 방법을 안내합니다.

### 1. 사전 요구사항

- Python 3.8 이상
- `pip` (Python 패키지 관리자)

### 2. 설치 및 설정

**1. 프로젝트 복제**
```bash
git clone https://your-repository-url.git
cd IN_GPS_SERVER
```

**2. 가상 환경 생성 및 활성화 (권장)**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. 의존성 패키지 설치**
프로젝트에 필요한 모든 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

**4. 환경 변수 설정**
`.env.example` 파일을 `.env` 파일로 복사한 후, 내용을 수정합니다. 이 프로젝트는 Django의 `SECRET_KEY`만 필요합니다.
```bash
cp .env.example .env
```

`.env` 파일 내용을 아래와 같이 수정하세요.
```dotenv
# Django Settings
# 이 값은 반드시 실제 운영 환경에서는 예측 불가능한 강력한 값으로 변경해야 합니다.
SECRET_KEY='your-django-secret-key-here'

# 디버그 모드 (개발 시 True, 운영 시 False)
DEBUG=True
```

**5. YOLO 모델 파일 준비**
`model_path/` 디렉터리 안에 분석에 사용할 `best_baseball_ball.pt` 파일을 위치시켜야 합니다. 이 파일은 Git에 포함되어 있지 않으므로 별도로 준비해야 합니다.

### 3. 서버 실행

모든 설정이 완료되면, 아래 명령어로 서버를 실행합니다. 이 스크립트는 **데이터베이스 파일(`db.sqlite3`)이 없으면 자동으로 생성 및 마이그레이션**한 후, `waitress` 프로덕션 서버를 `http://127.0.0.1:8000` 주소로 실행합니다.

```bash
python run_server.py
```

서버가 성공적으로 실행되면, 터미널에 다음과 같은 메시지가 출력됩니다.
```
데이터베이스 초기 설정을 시작합니다...
데이터베이스 설정이 완료되었습니다.
서버를 http://127.0.0.1:8000 에서 시작합니다.
Serving on http://127.0.0.1:8000
```

---

## API 문서

제공되는 모든 API의 상세한 명세(엔드포인트, 요청/응답 형식, 예시 등)는 아래 문서를 참조하세요.

- **[API_DOC.md](./API_DOC.md)**

---

## 프로젝트 구조

```
IN_GPS_SERVER/
├── analysis/          # 핵심 분석 기능 앱
│   ├── models.py      # 데이터 모델 (Player, VideoAnalysis 등)
│   ├── views.py       # API 로직
│   ├── urls.py        # API URL 라우팅
│   └── utils.py       # 영상 분석 알고리즘
├── IN_GPS_SERVER/     # Django 프로젝트 설정
│   ├── settings.py    # 메인 설정 (DB, 미디어 등)
│   └── wsgi.py        # WSGI 서버 연동
├── media/             # (자동생성) 업로드된 영상 파일 저장
├── model_path/        # YOLO 모델 파일 위치
├── .env               # (생성필요) 환경 변수 파일
├── .env.example       # 환경 변수 템플릿
├── API_DOC.md         # API 상세 명세서
├── run_server.py      # DB 자동 생성 및 서버 실행 스크립트
├── manage.py          # Django 관리 스크립트
├── requirements.txt   # 의존성 패키지 목록
└── README.md          # 프로젝트 안내 문서
```

---

## 라이선스

이 프로젝트는 `analysis/utils.py` 파일을 제외하고 [MIT License](./LICENSE)를 따릅니다.

`analysis/utils.py` 파일에 포함된 알고리즘은 **TACTICS**의 자산으로, 해당 파일 상단에 명시된 저작권 정책에 따라 상업적 사용, 수정 및 배포가 금지됩니다.
