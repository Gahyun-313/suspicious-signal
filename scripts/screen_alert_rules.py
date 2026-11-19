"""현재 농도와 회귀 예측을 결합한 경보 규칙을 선별한다.

공식 경보 라벨은 t+2 농도가 원점 ``conc_base_avg``의 1.3배 이상인지로
정의된다. 현재 농도 지속 신호와 모델의 t+2 농도 예측을 로그 비율로 결합해
Macro-F1과 최종 점수가 함께 개선되는 전역 규칙을 찾는다.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "KOWAS-EPI_전체데이터_v2.5" / "KOWAS-EPI"
PREDICTIONS = ROOT / "outputs" / "q1-improved-v1" / "predictions.csv"
SCORES = ROOT / "outputs" / "q1-improved-v1" / "score.json"
OUT = ROOT / "docs" / "internal" / "alert-rule-screen.csv"


def prepare_rows() -> pd.DataFrame:
    """평가 행에 원점 현재 농도와 경보 기준선을 결합한다."""

    panel = pd.read_csv(DATA / "kowas_epi_panel.csv", dtype={"region_code": str})
    predictions = pd.read_csv(PREDICTIONS)
    keys = pd.read_csv(DATA / "eval_keys.csv")

    # 모델과 같은 방식으로 원점 농도 결측을 지역별 최근 관측값으로 대체한다.
    panel = panel.sort_values(["region", "week_start_date"])
    panel["current_reference"] = panel.groupby("region")["conc_mean"].ffill().fillna(0.0)
    origins = panel.rename(columns={"date_week": "origin_week", "target_week_t2": "date_week"})
    rows = keys.merge(predictions, on=["region", "date_week"], validate="one_to_one")
    return rows.merge(origins, on=["region", "date_week"], validate="one_to_one")


def evaluate(rows: pd.DataFrame, prediction_weight: float, threshold: float) -> dict:
    """경보 규칙 한 조합의 10개 창 Macro-F1과 총점을 계산한다."""

    candidate = rows.dropna(subset=["target_alert_t2", "conc_base_avg"]).copy()
    # 비율을 로그 공간에서 섞으면 한 신호의 극단값이 곱셈적으로 지배하는 것을 줄인다.
    safe_base = candidate["conc_base_avg"].clip(lower=1e-9)
    current_ratio = np.log((candidate["current_reference"] + 1.0) / (safe_base + 1.0))
    prediction_ratio = np.log((candidate["pred_conc"] + 1.0) / (safe_base + 1.0))
    combined = prediction_weight * prediction_ratio + (1.0 - prediction_weight) * current_ratio
    candidate["candidate_alert"] = (combined >= np.log(threshold)).astype(int)

    # 회귀 항은 고정되어 있으므로 원래 예측의 분류만 바꿔 공식 점수를 재계산한다.
    fold_results = []
    for fold_id, fold in candidate.groupby("fold_id"):
        original = rows[rows["fold_id"] == fold_id].dropna(subset=["target_alert_t2"])
        original_f1 = f1_score(
            original["target_alert_t2"].astype(int), original["pred_alert"].astype(int), average="macro"
        )
        candidate_f1 = f1_score(
            fold["target_alert_t2"].astype(int), fold["candidate_alert"], average="macro"
        )
        # score.json의 구조를 직접 다루지 않도록 공식 회귀 행으로 회귀 항을 재계산한다.
        regression = rows[(rows["fold_id"] == fold_id)].dropna(
            subset=["target_conc_t2", "conc_3wk_avg"]
        )
        denominator = regression["pred_conc"].abs() + regression["target_conc_t2"].abs()
        model_smape = np.where(
            denominator == 0,
            0.0,
            200.0 * (regression["pred_conc"] - regression["target_conc_t2"]).abs() / denominator,
        ).mean()
        baseline_denominator = regression["conc_3wk_avg"].abs() + regression["target_conc_t2"].abs()
        baseline_smape = np.where(
            baseline_denominator == 0,
            0.0,
            200.0
            * (regression["conc_3wk_avg"] - regression["target_conc_t2"]).abs()
            / baseline_denominator,
        ).mean()
        score = 0.6 * (1.0 - model_smape / baseline_smape) + 0.4 * candidate_f1
        fold_results.append((fold_id, original_f1, candidate_f1, score))

    result = pd.DataFrame(
        fold_results, columns=["fold_id", "original_f1", "candidate_f1", "score"]
    )
    q1 = result[result["fold_id"].isin(["2025-Q1", "2026-Q1"])]
    return {
        "prediction_weight": prediction_weight,
        "threshold": threshold,
        "final_score": result["score"].mean(),
        "minimum_fold_score": result["score"].min(),
        "mean_macro_f1": result["candidate_f1"].mean(),
        "f1_gain": (result["candidate_f1"] - result["original_f1"]).mean(),
        "q1_macro_f1": q1["candidate_f1"].mean(),
    }


def main() -> None:
    if not PREDICTIONS.is_file() or not SCORES.is_file():
        raise FileNotFoundError("q1-improved-v1 공식 평가 결과가 필요합니다")

    rows = prepare_rows()
    grid = product(
        np.round(np.arange(0.0, 1.01, 0.1), 2),
        np.round(np.arange(0.8, 1.51, 0.05), 2),
    )
    results = pd.DataFrame(evaluate(rows, *parameters) for parameters in grid)
    results = results.sort_values(["final_score", "minimum_fold_score"], ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(results.head(20).to_string(index=False))
    print(f"전체 결과: {OUT}")


if __name__ == "__main__":
    main()
