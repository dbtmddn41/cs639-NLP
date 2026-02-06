# 실험 결과 요약

아래는 **원래 코드**와 **개선 코드**의 SST/CF-IMDB 성능 기록용 표입니다. 아직 측정값이 없으면 `N/A`로 두고, 실험 후 값으로 갱신하세요.

## 데이터셋별 성능

| 모델 | SST Dev Acc | SST Test Acc | CF-IMDB Dev Acc | CF-IMDB Test Acc |
|---|---:|---:|---:|---:|
| 원래 코드 | 0.3978 | 0.4253 | 0.9347 | 0.5123 |
| 개선 코드 v1 | 0.4187 | 0.4199 | 0.8898 | 0.5922 |
| 개선 코드 v2 (avgmax pool, layernorm, residual) | 0.4114 | 0.4421 | 0.8449 | 0.6025 |

## 실행 설정

- 원래 코드 실행 스크립트: 기본 `run_exp.sh`
- 개선 코드 v1 실행 스크립트: `9089214606/run_exp.sh` (AdamW, ReduceLROnPlateau, Early Stopping)
- 개선 코드 v2 실행 스크립트: `9089214606/run_exp.sh` (v1 + avgmax pooling, LayerNorm, residual connections)
- 사전학습 임베딩: GloVe 6B 300d
- 주요 개선사항 (v2):
  - **Pooling**: avg+max concatenation (특징 다양성)
  - **LayerNorm**: pooling 후, FFN 각 층 후 적용
  - **Residual connections**: FFN 층 간 연결
  - **드롭아웃 완화**: word_drop 0.3→0.1, emb_drop 0.333→0.1, hid_drop 0.333→0.3
  - **Label Smoothing**: SST 0.05, CF-IMDB 0.03
