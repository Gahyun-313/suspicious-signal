"""수준 모델과 로그 변화량 모델의 전역 혼합비를 선별한다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "KOWAS-EPI_전체데이터_v2.5" / "KOWAS-EPI"
LEVEL = ROOT / "outputs" / "asos-v1" / "predictions.csv"
CHANGE = ROOT / "outputs" / "log-change-only" / "predictions.csv"
OUT = ROOT / "docs" / "internal" / "change-blend-screen.csv"


def smape(prediction: pd.Series, actual: pd.Series) -> float:
    """공식 평가기와 같은 백분율 sMAPE를 계산한다."""

    denominator = prediction.abs() + actual.abs()
    values = np.where(denominator == 0, 0.0, 200.0 * (prediction - actual).abs() / denominator)
    return float(np.mean(values))


def prepare_rows() -> pd.DataFrame:
    """두 모델 예측과 공식 평가 행의 원점 정보를 결합한다."""

    panel = pd.read_csv(DATA / "kowas_epi_panel.csv", dtype={"region_code": str})
    keys = pd.read_csv(DATA / "eval_keys.csv")
    level = pd.read_csv(LEVEL).rename(columns={"pred_conc": "level_conc"})
    change = pd.read_csv(CHANGE).rename(columns={"pred_conc": "change_conc"})

    # 경보는 혼합 농도에서 다시 계산하므로 두 예측 파일의 기존 pred_alert는 버린다.
    level = level.drop(columns=["pred_alert"])
    change = change.drop(columns=["pred_alert"])
    rows = keys.merge(level, on=["region", "date_week"], validate="one_to_one")
    rows = rows.merge(change, on=["region", "date_week"], validate="one_to_one")

    panel = panel.sort_values(["region", "week_start_date"])
    panel["current_reference"] = panel.groupby("region")["conc_mean"].ffill().fillna(0.0)
    origins = panel.rename(columns={"date_week": "origin_week", "target_week_t2": "date_week"})
    return rows.merge(origins, on=["region", "date_week"], validate="one_to_one")


def evaluate(rows: pd.DataFrame, change_weight: float) -> dict:
    """변화량 모델 비중 하나를 공식 10개 창에서 재채점한다."""

    candidate = rows.copy()
    # 농도는 배율 차이가 크므로 원 스케일 평균이 아닌 로그 공간에서 섞는다.
    blended_log = (
        (1.0 - change_weight) * np.log1p(candidate["level_conc"].clip(lower=0.0))
        + change_weight * np.log1p(candidate["change_conc"].clip(lower=0.0))
    )
    candidate["pred_conc"] = np.expm1(blended_log).clip(lower=0.0)

    # 현재 채택 모델과 같은 회귀 연계 경보를 혼합 농도로 다시 계산한다.
    safe_base = candidate["conc_base_avg"].clip(lower=1e-9)
    predicted_ratio = np.log((candidate["pred_conc"] + 1.0) / (safe_base + 1.0))
    current_ratio = np.log((candidate["current_reference"] + 1.0) / (safe_base + 1.0))
    candidate["pred_alert"] = (0.9 * predicted_ratio + 0.1 * current_ratio >= np.log(1.1)).astype(int)

    fold_results = []
    regression_wins = 0
    for fold_id, fold in candidate.groupby("fold_id"):
        regression = fold.dropna(subset=["target_conc_t2", "conc_3wk_avg"])
        classification = fold.dropna(subset=["target_alert_t2"])
        model_smape = smape(regression["pred_conc"], regression["target_conc_t2"])
        baseline_smape = smape(regression["conc_3wk_avg"], regression["target_conc_t2"])
        macro_f1 = f1_score(
            classification["target_alert_t2"].astype(int),
            classification["pred_alert"].astype(int),
            average="macro",
        )
        score = 0.6 * (1.0 - model_smape / baseline_smape) + 0.4 * macro_f1
        fold_results.append((fold_id, score, model_smape, macro_f1))
        regression_wins += int(model_smape < baseline_smape)

    result = pd.DataFrame(
        fold_results, columns=["fold_id", "score", "model_smape", "macro_f1"]
    )
    q1 = result[result["fold_id"].isin(["2025-Q1", "2026-Q1"])]
    return {
        "change_weight": change_weight,
        "final_score": result["score"].mean(),
        "minimum_fold_score": result["score"].min(),
        "q1_mean_score": q1["score"].mean(),
        "mean_smape": result["model_smape"].mean(),
        "mean_macro_f1": result["macro_f1"].mean(),
        "regression_wins": regression_wins,
    }


def main() -> None:
    if not LEVEL.is_file() or not CHANGE.is_file():
        raise FileNotFoundError("수준 모델과 변화량 모델의 전체 예측 파일이 필요합니다")

    rows = prepare_rows()
    results = pd.DataFrame(
        evaluate(rows, weight) for weight in np.round(np.arange(0.0, 1.01, 0.05), 2)
    ).sort_values(["final_score", "minimum_fold_score"], ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(results.to_string(index=False))
    print(f"전체 결과: {OUT}")


if __name__ == "__main__":
    main()
