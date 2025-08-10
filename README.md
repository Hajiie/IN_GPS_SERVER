# IN_GPS_SERVER - 투수 분석 백엔드

Django 기반 투수 분석 백엔드 서버입니다.

## 시작하기

### 1. 환경 설정

#### 환경 변수 설정
1. `.env.example` 파일을 `.env`로 복사합니다:
```bash
 cp .env.example .env
```

2. `.env` 파일을 편집하여 실제 값들을 입력합니다:
```bash
 # Django Settings
SECRET_KEY=your-actual-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Settings
DB_NAME=ingps
DB_USER=ingpsuser
DB_PASSWORD=your-actual-database-password
DB_HOST=localhost
DB_PORT=5432
```

### 2. 의존성 설치
```bash
 pip install -r requirements.txt
```

### 3. 데이터베이스 마이그레이션
```bash
 python manage.py makemigrations
 python manage.py migrate
```

### 4. 서버 실행
```bash
 python manage.py runserver
```

## 보안 설정

### 중요: 환경 변수 관리
- **절대 `.env` 파일을 Git에 커밋하지 마세요!**
- `.env` 파일은 이미 `.gitignore`에 포함되어 있습니다
- 실제 운영 환경에서는 더 강력한 비밀번호를 사용하세요

### 환경 변수 목록
- `SECRET_KEY`: Django 시크릿 키 (운영 환경에서는 반드시 변경)
- `DEBUG`: 디버그 모드 (운영 환경에서는 False)
- `ALLOWED_HOSTS`: 허용된 호스트 목록
- `DB_NAME`: 데이터베이스 이름
- `DB_USER`: 데이터베이스 사용자
- `DB_PASSWORD`: 데이터베이스 비밀번호
- `DB_HOST`: 데이터베이스 호스트
- `DB_PORT`: 데이터베이스 포트

## 📁 프로젝트 구조

```
IN_GPS_SERVER/
├── analysis/          # 분석 앱
│   ├── models.py      # 데이터 모델
│   ├── views.py       # API 뷰
│   ├── urls.py        # URL 라우팅
│   └── utils.py       # 분석 유틸리티
├── IN_GPS_SERVER/     # 프로젝트 설정
│   ├── settings.py    # Django 설정
│   └── urls.py        # 메인 URL 설정
├── media/             # 업로드된 파일들
├── model_path/        # YOLO 모델 파일 (Git 제외)
├── ref_doc/           # 참고 문서 (Git 제외)
├── .env.example       # 환경 변수 템플릿
├── .gitignore         # Git 제외 파일 목록
├── API_DOC.md         # API 문서
└── README.md          # 이 파일
```

## API 문서

자세한 API 문서는 `API_DOC.md`를 참조하세요.

## 주의사항

1. **보안**: `.env` 파일에 실제 비밀번호와 API 키를 저장하세요
2. **데이터베이스**: 운영 환경에서는 PostgreSQL을 권장합니다
3. **파일 업로드**: `media/` 폴더는 Git에 포함되지 않습니다
4. **모델 파일**: `model_path/` 폴더의 YOLO 모델은 Git에 포함되지 않습니다

## 문제 해결

### 환경 변수 관련 오류
- `.env` 파일이 존재하는지 확인
- 환경 변수 이름이 올바른지 확인
- 값에 특수문자가 있다면 따옴표로 감싸기

### 데이터베이스 연결 오류
- PostgreSQL이 실행 중인지 확인
- 데이터베이스 사용자와 비밀번호가 올바른지 확인
- 데이터베이스가 생성되어 있는지 확인
