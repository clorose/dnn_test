# 실행 방법

## 1. 도커 설치

- 알아서..

## 2. 도커 컴포즈 빌드 및 실행

```bash
docker compose up --build dev
```

최초 빌드 이후엔 다음 명령어로 실행 가능합니다.

```bash
docker compose up dev
```


## 3. 도커 진입

```bash
docker exec -it "컨테이너 이름" zsh
```

## 4. 실행

```bash
cd src/
python3 "파일 이름"
```

## 5. 종료

Ctrl + D 로 도커 컨테이너에서 나온 후

```bash
docker compose down dev
```