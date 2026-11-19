# 🚨 수상한 시그널

![Python](https://img.shields.io/badge/Python_3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn_1.3-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy_1.26-013243?style=flat-square&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas_2.1-150458?style=flat-square&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Competition-Completed-2E7D32?style=flat-square)

**하수 감시 데이터로 지역별 2주 뒤 바이러스 농도와 급증 경보를 예측하는 시계열 AI 프로젝트**입니다.
미래 정보가 섞이지 않는 롤링 백테스트를 설계하고, 농도 수준·변화량 모델과 기상 특성을 결합해
공식 나이브 모델 대비 정량점수를 약 29.7% 개선했습니다.

> 공식 평가기 10개 롤링 창 기준: `0.2212 → 0.2870`

<br>

## 📖 프로젝트 소개

하수 속 바이러스 농도는 임상 신고보다 먼저 변할 수 있어 감염병 유행의 선행 신호로 활용할 수 있습니다.
이 프로젝트는 17개 시·도의 주간 하수 감시 자료, 전국 임상·농도 자료, ASOS 기상 관측을 이용해
각 예측 원점에서 **2주 뒤 농도와 급증 여부**를 함께 예측합니다.

```text
공식 평가기
  │  원점 이후 데이터와 라벨 차단
  ▼
누수 방지 FeatureBuilder ── ASOS 지역·주차 집계
  │
  ├─ 농도 수준 HGB 회귀 모델
  └─ 로그 변화량 HGB 회귀 모델
           │
           ▼
     로그 공간 앙상블
           │
           ├─ pred_conc  2주 뒤 농도
           └─ pred_alert 급증 경보 0/1
```

| 구현 항목 | 상태 |
|---|---|
| 데이터 구조·결측·이상치 분석 | ✅ 완료 |
| 시계열 누수 방지 특성 생성 | ✅ 완료 |
| 수준·변화량 이중 회귀 모델 | ✅ 완료 |
| ASOS 관측 기상 집계 | ✅ 완료 |
| 공식 10폴드 롤링 백테스트 | ✅ 완료 |
| 제출 패키지 자동 생성·검증 | ✅ 완료 |

<br>

## 📌 프로젝트 정보

| 항목 | 내용 |
|---|---|
| 대회 | [2026 KISTI DATA·AI 분석 경진대회](https://aida.kisti.re.kr/competition/main/competition/COMP_000000000000050/detail.do) |
| 과제 | 하수 감시 데이터 기반 감염병 유행 조기예측 AI 모델 개발 |
| 팀명 | 수상한 알파카 |
| 기간 | 2026.11 |
| 예측 대상 | 시·도별 2주 뒤 하수 바이러스 농도, 급증 경보 |
| 검증 방식 | 공식 평가기 10개 롤링 시계열 창 |
| 결과 | 로컬 공식 평가기 정량점수 **0.2870** |

> 이 저장소의 점수는 배포 데이터와 공식 평가기를 이용한 로컬 재현 결과입니다.
> 대회 최종 순위나 이후 기간의 일반화 성능을 의미하지 않습니다.

<br>

## 👨‍💻 구현 범위

- ZIP 내부 데이터·설명 문서 분석과 17개 지역 패널 프로파일링
- 목표 시점 이후 행·라벨·보조자료를 차단하는 롤링 검증 구성
- 지역 농도 지연값, 다중 기간 통계, 전국 신호, 계절성 특성 설계
- ASOS 일자료의 품질 규칙을 반영한 시·도-주차 집계
- HistGradientBoosting 기반 수준 모델과 로그 변화량 모델 구현
- 농도 예측과 기준 농도를 연결한 급증 경보 규칙 보정
- LightGBM·CatBoost·직접 분류·최근성 가중 등 후보 비교
- 예측 CSV, 자가 채점 결과, 실행 매뉴얼을 포함한 제출 ZIP 자동 생성

<br>

## 🛠️ 기술 스택

| 분류 | 기술 | 활용 |
|---|---|---|
| Language | Python 3.9 | 데이터 처리, 학습, 평가 자동화 |
| ML | scikit-learn 1.3.2 | HistGradientBoostingRegressor, Pipeline |
| Data | NumPy 1.26.1, pandas 2.1.2 | 특성 계산, 분석 리포트 |
| Candidate | LightGBM, CatBoost | 동일 검증 조건의 모델 계열 비교 |
| Validation | 공식 `evaluation.py` | 10개 롤링 창 재학습·예측·채점 |
| Documentation | Markdown, ReportLab | 실행 문서와 제출 보고서 생성 |

<br>

## ✨ 핵심 구현

### 1. 미래 정보를 차단한 롤링 검증

시계열 문제에서 무작위 분할은 미래 데이터가 과거 학습에 섞일 수 있습니다.
공식 평가기가 각 창의 `train_until`까지만 전달하도록 하고, 모델도 전달받은 `rows`, `national`,
`aux`만 사용하게 구성했습니다. 학습과 예측 단계에서 원본 전체 CSV를 직접 읽지 않습니다.

[아키텍처와 데이터 경계 자세히 보기 ↗](docs/ARCHITECTURE.md)

### 2. 결측과 큰 농도 편차를 고려한 특성

농도 범위가 `2.58 ~ 50,824,297.61`로 매우 넓어 농도·인구에 `log1p`를 적용했습니다.
결측은 학습 창 중앙값으로 대체하되 결측 여부 자체도 특성으로 남겼습니다.
현재부터 8주 전까지의 지연값과 최근 3·4·6·8·12주의 수준·변동성·관측률을 함께 사용했습니다.

### 3. 수준과 변화량을 함께 예측

절대 농도를 예측하는 수준 모델만으로는 갑작스러운 상승·하락을 포착하기 어려웠습니다.
별도 모델이 `log(t+2 농도) - log(현재 농도)`를 예측하게 하고, 두 결과를 로그 공간에서
90:10으로 결합했습니다. 변화량 모델 단독 성능은 낮았지만 보조 신호로 사용할 때
전체 점수가 `0.2833 → 0.2864`로 개선됐습니다.

### 4. 경보 규칙을 회귀 결과와 연결

작은 표본으로 급증 경보를 직접 분류하는 모델은 일반화 성능이 낮았습니다.
대신 최종 예측 농도와 현재 농도를 기준 농도와 비교해 경보를 계산했습니다.
예측 농도 80%, 현재 농도 20%, 임계값 1.09를 전체 평가 창에 동일하게 적용해
최종 점수를 `0.2864 → 0.2870`으로 개선했습니다.

<br>

## 📊 검증 결과

공식 `evaluation.py`가 평가 창마다 모델을 새로 학습하는 조건에서 측정했습니다.

| 단계 | 정량점수 | 변화 |
|---|---:|---:|
| 공식 나이브 모델 | 0.2212 | - |
| HGB 수준 모델 + 현재값 블렌딩 | 0.2616 | +0.0404 |
| 안전한 후행 평균 적용 | 0.2662 | +0.0046 |
| ASOS 기상 특성 추가 | 0.2833 | +0.0171 |
| 로그 변화량 앙상블 | 0.2864 | +0.0031 |
| 경보 규칙 재보정 | **0.2870** | +0.0006 |

- 공식 나이브 모델 대비 절대 `+0.0658`, 상대 약 **29.7%** 개선
- 최종 평가 키 2,159개, 누락·중복 0건
- 최종 예측의 NaN·Inf·음수 농도 및 잘못된 경보값 0건

평가 산식과 모델의 적용 범위·한계는 [모델 카드](docs/MODEL_CARD.md)에 정리했습니다.

<br>

## 🚀 시작하기

### 요구 사항

- Python 3.9
- 대회에서 배포한 `KOWAS-EPI_전체데이터_v2.5`

> 대회 원본 데이터는 이용 조건과 용량 문제로 Git에 포함하지 않습니다.

### 설치

```bash
git clone https://github.com/Gahyun-313/suspicious_signal.git
cd suspicious_signal
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 데이터 배치

```text
data/raw/KOWAS-EPI_전체데이터_v2.5/KOWAS-EPI/
├── evaluation.py
├── kowas_epi_panel.csv
├── national_weekly.csv
├── asos_stations.csv
└── ...
```

### 공식 백테스트

```powershell
$dataDir = '.\data\raw\KOWAS-EPI_전체데이터_v2.5\KOWAS-EPI'
$projectDir = (Get-Location).Path
Push-Location $dataDir
python .\evaluation.py selftest
python .\evaluation.py backtest `
  --model "$projectDir\src\model.py" `
  --out "$projectDir\outputs\final\predictions.csv" `
  --json "$projectDir\outputs\final\score.json"
Pop-Location
```

정상 실행 시 공식 10개 롤링 평가 창의 최종 점수 약 `0.2870`이 출력됩니다.

### 데이터 분석 보고서 재생성

```powershell
python .\scripts\analyze_data.py
```

### 제출 문서 재생성

```powershell
python -m pip install -r requirements-docs.txt
python .\scripts\build_submission.py
```

<br>

## 📁 프로젝트 구조

```text
.
├── src/
│   ├── model.py              수준·변화량 모델, 앙상블, 경보
│   ├── feature_builder.py    누수 방지 시계열 특성 생성
│   └── asos_features.py      ASOS 일자료 지역·주차 집계
├── scripts/
│   ├── analyze_data.py       데이터 프로파일 생성
│   ├── analyze_errors.py     평가 창·지역별 오차 분석
│   └── build_submission.py   제출 문서·모델 ZIP 생성
├── experiments/              채택·미채택 모델 비교 코드
├── submission/               공개 가능한 보고서·실행 매뉴얼 원본
├── docs/                     아키텍처, 모델 카드, 데이터 프로파일
└── requirements.txt          재현 환경 의존성
```

`data/`, `outputs/`, `dist/`, `docs/internal/`은 Git 추적에서 제외합니다.

<br>

## 📚 문서

| 문서 | 내용 |
|---|---|
| [아키텍처](docs/ARCHITECTURE.md) | 학습·예측 흐름, 누수 방지 경계, 구성 요소 |
| [모델 카드](docs/MODEL_CARD.md) | 입력·출력, 평가 지표, 성능, 한계 |
| [데이터 프로파일](docs/data-profile.md) | 데이터 구조, 결측, 이상 신호 |
| [문제해결 방법 설명서](submission/document/solution_report.md) | 제출 당시 모델 개발·평가 보고서 |
| [모델 실행 매뉴얼](submission/model/RUN_MANUAL.md) | 평가 환경 설치와 재현 명령 |

<br>

## 🔒 공개 범위

- 대회 원본 데이터와 평가 산출물은 저장소에 포함하지 않습니다.
- 개발 중 생성된 비공개 실험 노트는 `docs/internal/`에 보관하며 Git에서 제외합니다.
- 공개된 점수는 공식 배포 데이터에서 재현한 로컬 결과만 기재합니다.
