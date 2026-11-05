# 수상한 시그널

2026 KISTI DATA·AI 분석 경진대회 「하수 감시 데이터 기반 감염병 유행
조기예측 AI 모델 개발」 참가 프로젝트입니다.

- 팀명: 수상한 알파카
- 목표: 누수 없는 공식 롤링 백테스트에서 재현 가능한 성능 개선
- 원본 데이터: `data/raw/` (Git 추적 제외)
- 공식 기준 데이터: `KOWAS-EPI_전체데이터_v2.5`

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
