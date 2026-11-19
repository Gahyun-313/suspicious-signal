"""기존 예측으로 적응형 블렌딩 후보를 빠르게 선별한다.

현재 모델의 최종 로그 예측은 ``0.4 * 학습 모델 + 0.6 * 현재 농도``다.
이 관계에서 학습 모델 단독 예측을 복원한 뒤, 두 예측의 차이가 클수록 학습
모델의 비중을 높이는 후보를 재채점한다. 실제 모델 재학습 전의 선별 도구이며
결과는 Git에서 제외된 ``docs/internal``에만 기록한다.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "KOWAS-EPI_전체데이터_v2.5" / "KOWAS-EPI"
PREDICTIONS = ROOT / "outputs" / "baseline-v1" / "predictions.csv"
OUT = ROOT / "docs" / "internal" / "adaptive-blend-screen.csv"
ORIGINAL_ALPHA = 0.4


def smape(prediction: pd.Series, actual: pd.Series) -> float:
    """공식 평가기와 같은 백분율 sMAPE를 계산한다."""

    denominator = prediction.abs() + actual.abs()
    values = np.where(denominator == 0, 0.0, 200.0 * (prediction - actual).abs() / denominator)
    return float(np.mean(values))


def prepare_rows() -> pd.DataFrame:
    """평가 행에 원점의 현재값 대체치와 기존 예측을 결합한다."""

    panel = pd.read_csv(DATA / "kowas_epi_panel.csv", dtype={"region_code": str})
    predictions = pd.read_csv(PREDICTIONS)
    keys = pd.read_csv(DATA / "eval_keys.csv")

    # 실제 모델과 똑같이 원점 결측을 같은 지역의 최근 유효 농도로 대체한다.
    panel = panel.sort_values(["region", "week_start_date"])
    panel["current_reference"] = panel.groupby("region")["conc_mean"].ffill().fillna(0.0)
    # 배포 conc_3wk_avg의 알려진 미래값 혼입 1행을 피하기 위해 원점 이하
    # conc_mean만으로 후행 3개 주의 산술평균을 다시 만든다.
    panel["safe_3wk_avg"] = panel.groupby("region")["conc_mean"].transform(
        lambda values: values.rolling(window=3, min_periods=1).mean()
    )
    origins = panel.rename(columns={"date_week": "origin_week", "target_week_t2": "date_week"})
    rows = keys.merge(predictions, on=["region", "date_week"], validate="one_to_one")
    rows = rows.merge(origins, on=["region", "date_week"], validate="one_to_one")

    # 기존 최종 예측에서 학습 모델 단독 로그 예측을 역산한다.
    final_log = np.log1p(rows["pred_conc"].clip(lower=0.0))
    current_log = np.log1p(rows["current_reference"].clip(lower=0.0))
    rows["learned_log"] = (final_log - (1.0 - ORIGINAL_ALPHA) * current_log) / ORIGINAL_ALPHA
    rows["current_log"] = current_log
    rows["disagreement"] = (rows["learned_log"] - rows["current_log"]).abs()
    return rows


def evaluate(
    rows: pd.DataFrame,
    base_alpha: float,
    gain: float,
    max_alpha: float,
    baseline_weight: float,
) -> dict:
    """적응형 비중과 공식 기준선 혼합 한 조합을 10개 창에서 평가한다."""

    candidate = rows.copy()
    candidate["alpha"] = np.minimum(max_alpha, base_alpha + gain * candidate["disagreement"])
    blended_log = (
        candidate["alpha"] * candidate["learned_log"]
        + (1.0 - candidate["alpha"]) * candidate["current_log"]
    )
    # 공식 3주 평균이 있는 행은 로그 공간에서 전역 비율로 한 번 더 섞는다.
    # 결측인 행은 모델 예측을 그대로 사용해 임의의 0 대체를 피한다.
    baseline_log = np.log1p(candidate["safe_3wk_avg"].clip(lower=0.0))
    baseline_log = baseline_log.fillna(blended_log)
    final_log = (1.0 - baseline_weight) * blended_log + baseline_weight * baseline_log
    candidate["candidate_conc"] = np.expm1(final_log).clip(lower=0.0)

    fold_scores = []
    regression_wins = 0
    q1_scores = []
    for fold_id, fold in candidate.groupby("fold_id"):
        regression = fold.dropna(subset=["target_conc_t2", "conc_3wk_avg"])
        classification = fold.dropna(subset=["target_alert_t2"])
        model_smape = smape(regression["candidate_conc"], regression["target_conc_t2"])
        baseline_smape = smape(regression["conc_3wk_avg"], regression["target_conc_t2"])
        macro_f1 = f1_score(
            classification["target_alert_t2"].astype(int),
            classification["pred_alert"].astype(int),
            average="macro",
        )
        score = 0.6 * (1.0 - model_smape / baseline_smape) + 0.4 * macro_f1
        fold_scores.append(score)
        regression_wins += int(model_smape < baseline_smape)
        if fold_id in {"2025-Q1", "2026-Q1"}:
            q1_scores.append(score)

    return {
        "base_alpha": base_alpha,
        "gain": gain,
        "max_alpha": max_alpha,
        "baseline_weight": baseline_weight,
        "final_score": float(np.mean(fold_scores)),
        "minimum_fold_score": float(np.min(fold_scores)),
        "q1_mean_score": float(np.mean(q1_scores)),
        "regression_wins": regression_wins,
        "mean_alpha": float(candidate["alpha"].mean()),
        "p90_alpha": float(candidate["alpha"].quantile(0.9)),
    }


def main() -> None:
    if not PREDICTIONS.is_file():
        raise FileNotFoundError(f"기준 예측 파일이 없습니다: {PREDICTIONS}")

    rows = prepare_rows()
    # base=0.4, gain=0은 현재 모델을 그대로 재현하는 대조군이다.
    grid = product(
        [0.3, 0.4, 0.5],
        [0.0, 0.05, 0.10, 0.15, 0.20],
        [0.6, 0.7, 0.8, 0.9],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    results = [evaluate(rows, *parameters) for parameters in grid if parameters[0] <= parameters[2]]
    result = pd.DataFrame(results).sort_values(
        ["final_score", "minimum_fold_score"], ascending=False
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(result.head(15).to_string(index=False))
    print(f"전체 결과: {OUT}")


if __name__ == "__main__":
    main()
