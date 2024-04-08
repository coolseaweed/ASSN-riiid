# Riiid problem 3 

## 1. Training
### 1-1. 학습 모델
학습에 사용된 모델은 `LogisticRegression` 모델이며, 테이블 형태의 데이터를 classification 할때 쉽게 적용해볼 수 있을 것 같다고 판단했습니다.

### 1-2. 환경셋업
```
docker compose --profile train up -d --build 
```
### 1-3. EDA 및 학습
EDA 및 학습 과정은 [`./train/train.ipynb`](./train/train.ipynb) 파일을 통해 확인하실 수 있으며, 간단하게 아래 커맨드를 통해 모델 학습을 할수 있습니다.
```bash
docker exec riiid-p3-train-1 python train.py [--overwrite]
```
학습된 모델 weight 들은 다음과 같습니다.
```bash
./models/
├── label_encoder.pkl
├── model.pkl
└── one_hot_encoder.pkl
```
### 1-4. 성능 평가
주어진 데이터의 **30%** 를 테스트 셋으로 분리 후 학습했으며, **학습 셋**과 **테스트 셋**에 대한 정확도는 각각 **71%**, **66%** 로 측정되었습니다. 
|Adaptive Level|precision|recall|fi-score|
|:----|:---:|:---:|:----:|
|High|0.40|0.67|0.50|
|Moderate|0.67|0.66|0.66|
|Low|0.73|0.65|0.69|

## 2. Serving
### 2-1. 서빙 아키텍처

### 2-2. 환경 셋업
```
docker compose --profile train up -d --build 
```