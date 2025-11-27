**개요**

SpatioTemporal Attention기반 오존 예측 모델 코드베이스. 

**설치 및 환경**

- **Conda 환경 생성:** 저장소 루트의 `ozone_env.yml` 파일로 환경을 생성. 

```bash
conda env create -f ozone_env.yml
conda activate <env_name> 
```

**프로젝트 구조(요약)**
- **`Data/`**: netcdf형식의 파일을 npy형태로 가공
- **`Dataset/`**: 가공된 npy데이터를 이용한 pytorch 프레임워크의 데이터셋
- **`Models/`**: SpatioTemporal Attention이 적용된 Transformer기반 오존 예측모델
- **`ModelsWeights/`**: pth형식의 학습된 model의 weight 및 모델의 hyperparmeter configuration가 정의된 yaml 파일 (head수, encoder 및 decoer 수 등 정의)
- **`Vadliate/`**: 통계평가 및 지수평가를 위한 코드 및 결과
- **`Visualize/`**: 학습 결과 시각화 코드

**데이터**

- 데이터는 저장소에 포함되어 있지 않음. 
---