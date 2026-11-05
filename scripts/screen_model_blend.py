"""HGB와 LightGBM 최종 농도 예측의 전역 혼합비를 선별한다."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "KOWAS-EPI_전체데이터_v2.5" / "KOWAS-EPI"
HGB = ROOT / "outputs" / "level-change-ensemble-v1" / "predictions.csv"
LIGHTGBM = ROOT / "outputs" / "lightgbm-v1" / "predictions.csv"
OUT = ROOT / "docs" / "internal" / "model-blend-screen.csv"


def smape(prediction: pd.Series, actual: pd.Series) -> float:
    """공식 평가기와 같은 백분율 sMAPE를 계산한다."""

    denominator = prediction.abs() + actual.abs()
    values = np.where(denominator == 0, 0.0, 200.0 * (prediction - actual).abs() / denominator)
    return float(np.mean(values))


def prepare_rows() -> pd.DataFrame:
    """두 모델 예측과 원점 농도·경보 기준선을 평가 키로 결합한다."""

    panel = pd.read_csv(DATA / "kowas_epi_panel.csv", dtype={"region_code": str})
    keys = pd.read_csv(DATA / "eval_keys.csv")
    hgb = pd.read_csv(HGB).rename(columns={"pred_conc": "hgb_conc"}).drop(columns=["pred_alert"])
    lightgbm = (
        pd.read_csv(LIGHTGBM)
        .rename(columns={"pred_conc": "lightgbm_conc"})
        .drop(columns=["pred_alert"])
    )
    rows = keys.merge(hgb, on=["region", "date_week"], validate="one_to_one")
    rows = rows.merge(lightgbm, on=["region", "date_week"], validate="one_to_one")

    # 원점 결측은 두 실제 모델과 동일하게 지역별 최근 유효 농도로 대체한다.
    panel = panel.sort_values(["region", "week_start_date"])
    panel["current_reference"] = panel.groupby("region")["conc_mean"].ffill().fillna(0.0)
    origins = panel.rename(columns={"date_week": "origin_week", "target_week_t2": "date_week"})
    return rows.merge(origins, on=["region", "date_week"], validate="one_to_one")


def evaluate(rows: pd.DataFrame, lightgbm_weight: float) -> dict:
    """LightGBM 비중 하나를 공식 10개 창에서 재채점한다."""

    candidate = rows.copy()
    # 농도는 배율 차이가 크므로 두 모델의 예측을 로그 공간에서 혼합한다.
    final_log = (
        (1.0 - lightgbm_weight) * np.log1p(candidate["hgb_conc"].clip(lower=0.0))
        + lightgbm_weight * np.log1p(candidate["lightgbm_conc"].clip(lower=0.0))
    )
    candidate["pred_conc"] = np.expm1(final_log).clip(lower=0.0)

    # 혼합 농도에 현재 모델의 회귀 연계 경보 규칙을 동일하게 다시 적용한다.
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
        "lightgbm_weight": lightgbm_weight,
        "final_score": result["score"].mean(),
        "minimum_fold_score": result["score"].min(),
        "q1_mean_score": q1["score"].mean(),
        "mean_smape": result["model_smape"].mean(),
        "mean_macro_f1": result["macro_f1"].mean(),
        "regression_wins": regression_wins,
    }


def main() -> None:
    if not HGB.is_file() or not LIGHTGBM.is_file():
        raise FileNotFoundError("HGB와 LightGBM의 전체 예측 파일이 필요합니다")

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
