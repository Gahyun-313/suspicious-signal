# 수상한 시그널

2026 KISTI DATA·AI 분석 경진대회 「하수 감시 데이터 기반 감염병 유행
조기예측 AI 모델 개발」 참가 프로젝트입니다.

- 팀명: 수상한 알파카
- 목표: 누수 없는 공식 롤링 백테스트에서 재현 가능한 성능 개선
- 원본 데이터: `data/raw/` (Git 추적 제외)
- 공식 기준 데이터: `KOWAS-EPI_전체데이터_v2.5`

## 현재 검증 결과

공식 `evaluation.py`의 10개 롤링 평가 창을 그대로 사용한 결과입니다.

| 모델 | 최종 정량점수 |
|---|---:|
| 공식 나이브 기준 모델 | 0.2212 |
| 현재 베이스라인 | **0.2833** |

현재 모델은 원점 이하 자료와 공식 평가기가 `aux`로 전달한 ASOS 관측 기상만
사용하며, 배포 `conc_3wk_avg`의 알려진 미래값 혼입 행을 출력 계산에 직접
사용하지 않습니다. 점수는 로컬 공식 평가기에서 재현한 값이며 대회 최종
순위나 이후 기간의 일반화 성능을 의미하지 않습니다.

## 실행

공식 평가기는 배포 데이터 폴더에서 실행해야 합니다. PowerShell 기준:

```powershell
$dataDir = '.\data\raw\KOWAS-EPI_전체데이터_v2.5\KOWAS-EPI'
$projectDir = (Get-Location).Path
Push-Location $dataDir
python .\evaluation.py selftest
python .\evaluation.py backtest `
  --model "$projectDir\src\model.py" `
  --out "$projectDir\outputs\baseline-v1\predictions.csv" `
  --json "$projectDir\outputs\baseline-v1\score.json"
Pop-Location
```

데이터 점검 보고서는 다음 명령으로 다시 만들 수 있습니다.

```powershell
python .\scripts\analyze_data.py
```
