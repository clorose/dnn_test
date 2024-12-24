# CNC DNN 프로젝트 문서

## 프로젝트 소개
이 프로젝트는 CNC 가공 데이터를 활용하여 가공 품질을 예측하는 딥러닝 모델을 개발하고 최적화하는 것을 목표로 합니다. 

주요 기능:
- 가공 품질에 대한 이진 분류(정상/불량) 및 3클래스 분류(정상/부분불량/전체불량) 지원
- 베이지안 최적화를 통한 자동 하이퍼파라미터 튜닝
- 실험 결과의 체계적인 저장 및 관리
- 학습 과정 시각화 및 성능 지표 분석

## 프로젝트 구조
```
.
├── README.md                # 프로젝트 설명 문서
├── docker-compose.yml       # 도커 컨테이너 설정
├── src/                     # 소스 코드 디렉토리
│   ├── main.py             # 메인 실행 파일 (일반 실험용)
│   ├── config.py           # 설정 클래스 (실험 설정 관리)
│   ├── experiment.py       # 실험 클래스 (모델 학습 및 평가)
│   ├── optimizers.py       # 옵티마이저 클래스 (학습 최적화)
│   ├── bayesian_opt.py     # 베이지안 최적화 클래스
│   ├── data_processing.py  # 데이터 전처리 기능
│   ├── utils.py            # 유틸리티 함수 모음
│   └── convert_yaml.py     # YAML 파일 변환 도구
├── data/                    # 데이터 디렉토리
│   ├── train.csv           # 학습 데이터 메타정보
│   └── CNC Virtual Data set _v2/  # 실험 데이터 세트
├── runs/                    # 실험 결과 저장소
│   └── run_{timestamp}/    # 각 실험별 결과
└── experiments/            # 실험 설정 파일 보관
    ├── base.yaml          # 기본 실험 설정
    └── optimal/           # 최적화된 설정 파일들
```

## 설치 및 실행 환경 설정

### Docker 설치 가이드

#### Windows 사용자
1. WSL2(Windows Subsystem for Linux 2) 설치 필요
   - Windows 10 버전 2004 이상 필요
   - Microsoft Store에서 'Windows Terminal' 설치 권장
   - PowerShell 관리자 모드에서 WSL2 설치:
     ```powershell
     wsl --install
     ```

2. Docker Desktop 설치
   - [Docker Desktop 설치 페이지](https://docs.docker.com/desktop/install/)에서 Windows 버전 다운로드
   - 설치 시 'WSL2 기반 엔진' 옵션 선택
   - 설치 완료 후 재부팅

#### Linux/Mac 사용자
- **Ubuntu:**
  ```bash
  sudo apt-get update
  sudo apt-get install docker.io docker-compose
  ```
- **Mac:**
  - [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/) 설치
  - Apple Silicon/Intel CPU 버전 구분하여 설치

### 도커 환경 설정 상세 가이드

#### 컨테이너 빌드 및 실행
1. 최초 빌드 시:
   ```bash
   docker compose up --build dev
   ```
   - 모든 의존성 패키지 설치
   - 기본 환경 구성
   - 약 5-10분 소요될 수 있음

2. 이후 실행 시:
   ```bash
   docker compose up dev
   ```
   - 이미 빌드된 이미지 사용
   - 빠른 시작 가능

#### 컨테이너 접속 방법
1. 새 터미널 창 열기
2. 실행 중인 컨테이너 확인:
   ```bash
   docker compose ps
   ```
3. 컨테이너 접속:
   ```bash
   docker exec -it "컨테이너_이름" zsh
   ```
   - zsh: 향상된 쉘 환경 제공
   - 자동 완성, 히스토리 검색 등 기능 제공
   - 기본 테마 및 플러그인 설정 완료

#### 컨테이너 종료 상세 방법
1. 권장하는 종료 방법:
   ```bash
   docker compose down dev
   ```
   - 모든 관련 리소스 정리
   - 네트워크, 볼륨 정리
   - 메모리 완전 해제

2. 임시 종료 (Ctrl + C):
   - 컨테이너만 일시 중지
   - 리소스는 유지됨
   - 재시작 시 빠름
   - 완전한 종료를 위해서는 `docker compose down dev` 필요

## 실험 실행 상세 가이드

### 1. 일반 실험 실행
기본 설정 또는 사용자 정의 설정으로 모델을 학습시킬 수 있습니다.

```bash
cd src
python main.py --config "../experiments/base.yaml"
```

주요 매개변수:
- `--config`: 실험 설정 파일 경로
- `--data_path`: 데이터 디렉토리 경로 (기본값: "../data")

실험 과정:
1. 데이터 로드 및 전처리
2. 모델 구성 및 컴파일
3. 학습 수행
4. 결과 저장 및 시각화

### 2. 베이지안 최적화 실행
하이퍼파라미터 최적화를 통해 최적의 모델 설정을 찾습니다.

```bash
cd src
python run_bayesian_opt.py --config "../experiments/base.yaml" --n_trials 20
```

매개변수 설명:
- `--config`: 기본 설정 파일 경로
- `--n_trials`: 최적화 시도 횟수 (최소 10회 필수)
  - 최소 권장 : 20회
  - 적정 권장 : 40-50회
  - 최적 권장 : 100회
  - 시도 횟수가 많을수록 더 정확한 최적화 가능
  - 단 시간 소요 증가
- `--data_path`: 데이터 경로 (선택적)

최적화 대상 파라미터:
- 네트워크 구조 (레이어 크기, 개수)
- 학습률 및 배치 크기
- 드롭아웃 비율
- 정규화 강도
- 옵티마이저 설정

### 3. 최적화된 설정으로 실험 실행
최적화를 통해 찾은 최적의 파라미터로 실험을 수행합니다.

1. YAML 파일 정리 (필수):
   ```bash
   python convert_yaml.py "../experiments/optimal/optimal_params_{timestamp}.yaml"
   ```
   - 베이지안 최적화로 생성된 NumPy 객체를 기본 Python 타입으로 변환
   - 이 과정을 거치지 않으면 YAML 파일을 읽을 수 없음

2. 실험 실행:
   ```bash
   python main.py --config "../experiments/optimal/optimal_params_{timestamp}.yaml"
   ```

## 결과 분석 상세 설명

### 일반 실험 결과 (`runs/run_{timestamp}/`)
- `model.keras`: 
  - 저장된 모델 파일
  - Tensorflow 2.x 형식
  - 가중치 및 모델 구조 포함

- `metrics.json`: 
  - 정확도, 손실 값
  - 검증 성능
  - 에폭별 기록

- `learning_curves.png`:
  - 학습/검증 정확도 그래프
  - 학습/검증 손실 그래프
  - 과적합 진단 가능

- `config.json`:
  - 실험에 사용된 모든 설정
  - 하이퍼파라미터 기록
  - 재현성 보장

### 최적화 결과 (`experiments/optimal/`)
- `optimal_params_{timestamp}.yaml`:
  - 최적 하이퍼파라미터
  - 사람이 읽을 수 있는 형식
  - 직접 수정 가능

- `optimization_results_{timestamp}.json`:
  - 전체 최적화 과정 기록
  - 시도된 모든 파라미터
  - 각 시도의 성능 점수

## 기타 설정

### Shell 환경
- zsh 기반 환경 권장
  - 자동 완성
  - 구문 강조
  - 깃 상태 표시
- [p10k 설정 및 폰트](https://drive.google.com/drive/folders/1Upm2IqMFXcjj1bIKIsR7CUpoEo4Zrl9f?usp=sharing)
  - 커스텀 프롬프트
  - 아이콘 지원
  - 개발 편의성 향상

### 개발 참고사항
1. 코드 수정
   - `src/` 디렉토리가 볼륨으로 설정
   - 실시간 코드 반영 가능
   - 컨테이너 재시작 불필요

2. 패키지 관리
   - 새로운 패키지 필요시 `requirements.txt` 수정
   - 컨테이너 재빌드 필요
   - 버전 명시 권장

3. 결과 관리
   - 모든 실험 결과는 `runs/` 디렉토리에 자동 저장
   - 타임스탬프로 구분
   - 백업 권장

### 주의사항
1. 리소스 관리
   - 사용 후 컨테이너 종료 필수
   - 메모리 누수 방지
   - 시스템 성능 유지

2. 데이터 처리
   - 대용량 데이터 처리 시 충분한 메모리 확보
   - 데이터 전처리 상태 확인
   - 백업 유지

3. 최적화 실행
   - 충분한 시간 확보 필요
   - 리소스 사용량 모니터링
   - 중간 결과 저장 활용

4. 실험 재현성
   - 설정 파일 보관
   - 랜덤 시드 설정(42 는 모든 것을 해결함.)
   - 환경 의존성 주의