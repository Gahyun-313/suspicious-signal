# 수상한 시그널 모델 실행 매뉴얼

## 구성 파일

- `model.py`: 공식 평가기가 불러오는 `Model` 클래스
- `feature_builder.py`: 누수 방지 시계열 특성 생성
- `asos_features.py`: 공식 `aux` ASOS 일자료 집계
- `requirements.txt`: 재현 환경의 Python 패키지 버전

원본 대회 데이터와 평가기는 모델자료 ZIP에 포함하지 않는다.

## 권장 환경

- Windows 10/11 또는 Linux
- Python 3.9.13
- NumPy 1.26.1
- pandas 2.1.2
- scikit-learn 1.3.2

## 설치

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 공식 평가기 실행

모델자료 ZIP을 압축 해제한 폴더를 `$modelDir`, 공식 데이터 v2.5에서
`evaluation.py`가 있는 폴더를 `$dataDir`로 지정한다.

```powershell
$modelDir = (Resolve-Path '.\model_submission').Path
$dataDir = (Resolve-Path '.\KOWAS-EPI').Path
Push-Location $dataDir
python .\evaluation.py selftest
python .\evaluation.py backtest `
  --model "$modelDir\model.py" `
  --out "$modelDir\predictions.csv" `
  --json "$modelDir\score.json"
Pop-Location
```

정상 실행 시 공식 10개 롤링 평가 창의 최종 정량점수 약 `0.2870`이 출력된다.
실행 시간은 PC 환경에 따라 다르며 검증 환경에서는 약 4~6분이 소요됐다.

## 모델 인터페이스

`evaluation.py`는 각 평가 창마다 `Model`을 새로 생성하고 다음 메서드를 호출한다.

```python
model.fit(rows, national, aux=aux)
predictions = model.predict(rows, origin_week, national, aux=aux)
```

반환값은 `{지역명: (2주 뒤 예측 농도, 급증 경보 0 또는 1)}` 형식이다.
`KOWAS_STATIC_DIR` 환경변수와 `aux` 자료는 공식 평가기가 자동으로 전달한다.

## 재현성 및 주의사항

- 모델은 공식 평가기가 전달한 원점 이하 자료만 사용한다.
- `KOWAS_STATIC_DIR/asos_stations.csv` 외의 파일을 직접 읽지 않는다.
- 원본 데이터 파일을 모델 폴더로 복사할 필요가 없다.
- `model.py`, `feature_builder.py`, `asos_features.py`는 같은 폴더에 둔다.
- 패키지 버전이 다르면 미세한 예측 차이가 발생할 수 있으므로 고정 버전을 권장한다.
