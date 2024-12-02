# 도커 사용법

## 도커 설치

- **Windows** 에서 Docker 설치를 위해 WSL2 설치 필요
- 설치법은 [Docker Desktop 설치](https://docs.docker.com/desktop/install/) 문서 참고

## 도커 컴포즈 빌드 및 실행

- 현재 도커 컴포즈는 `dev` 환경을 사용하도록 설정되어 있음
- 이 환경에서는 `tail -f /dev/null` 명령어를 통해 컨테이너가 종료되지 않도록 설정되어 있음
- 최초 빌드 또는 변경사항이 있을 때는 `--build` 옵션을 사용하여 빌드해야 함

```bash
docker compose up --build dev
```

빌드 이후엔 다음 명령어로 실행 가능함:
```bash
docker compose up dev
```

## 도커 진입

- 현재 `tail -f /dev/null` 명령어로 컨테이너가 실행되고 있기 때문에 다른 터미널을 열어 진입해야 함
- 컨테이너 이름은 `docker compose ps` 명령어로 확인 가능함
- `zsh` 대신 `bash` 또는 `sh` 등을 사용할 수 있지만, `zsh`를 사용하는 것을 권장함

```bash
docker exec -it "컨테이너 이름" zsh
```

## 실행

컨테이너 진입 후:
1. `src/` 디렉토리로 이동하여 실행 (`cd src/`)
2. Python 파일 실행: `python3 "파일 이름"`

* 코드들은 `src/` 디렉토리에 위치함
* `docker-compose.yml` 파일에서 `src/` 디렉토리를 볼륨으로 설정했기 때문에 코드 변경이 자동으로 반영됨
* 일반 터미널처럼 사용 가능
* 컨테이너 쉘에서 나갈 때는 `Ctrl + D` 사용

## 컨테이너 종료

컨테이너를 종료하는 방법은 두 가지가 있음:

1. `docker compose down dev` 명령어 사용 (권장)
```bash
docker compose down dev
```

2. `Ctrl + C` 로 종료
   - 주의: 이 방법은 컨테이너가 완전히 종료되지 않을 수 있음
   - 완전한 종료를 위해서는 `docker compose down dev` 실행 권장

### 종료 이유
- 실행 중인 컨테이너는 시스템 리소스(메모리, CPU)를 계속 사용함
- 불필요한 리소스 사용을 방지하기 위해 사용하지 않을 때는 종료하는 것이 좋음

### 재시작
종료 후 다시 시작하려면:
```bash
docker compose up dev
```


## 참고 

[p10k 설정 및 폰트](https://drive.google.com/drive/folders/1Upm2IqMFXcjj1bIKIsR7CUpoEo4Zrl9f?usp=sharing)

