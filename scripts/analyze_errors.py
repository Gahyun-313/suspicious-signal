"""공식 백테스트 예측의 지역·주차별 오차를 분석한다.

결과는 ``docs/internal/``에만 저장한다. 이 폴더는 대회 전략과 상세 실패
기록이 공개 저장소에 포함되지 않도록 ``.gitignore``에서 제외되어 있다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "KOWAS-EPI_전체데이터_v2.5" / "KOWAS-EPI"
DEFAULT_PREDICTIONS = ROOT / "outputs" / "baseline-v1" / "predictions.csv"
REPORT = ROOT / "docs" / "internal" / "q1-error-analysis.md"
DETAIL = ROOT / "docs" / "internal" / "q1-error-detail.csv"


def smape_element(prediction: pd.Series, actual: pd.Series) -> pd.Series:
    """공식 평가식과 같은 행 단위 sMAPE 기여도(0~200)를 계산한다."""

    denominator = prediction.abs() + actual.abs()
    # 실제값과 예측값이 모두 0이면 공식 정의에 따라 오차를 0으로 둔다.
    return np.where(denominator == 0, 0.0, 200.0 * (prediction - actual).abs() / denominator)


def load_evaluation_rows(predictions_path: Path) -> pd.DataFrame:
    """평가 키에 해당하는 정답·예측·원점 특성을 한 테이블로 결합한다."""

    panel = pd.read_csv(DATA / "kowas_epi_panel.csv", dtype={"region_code": str})
    keys = pd.read_csv(DATA / "eval_keys.csv")
    predictions = pd.read_csv(predictions_path)

    # 제출 파일 date_week는 목표주(t+2)이고 패널에서는 target_week_t2에 대응한다.
    # 패널의 기존 date_week는 예측을 수행한 원점이므로 origin_week로 분리한다.
    evaluation = keys.merge(predictions, on=["region", "date_week"], how="left", validate="one_to_one")
    origin = panel.rename(columns={"date_week": "origin_week", "target_week_t2": "date_week"})
    evaluation = evaluation.merge(
        origin,
        on=["region", "date_week"],
        how="left",
        validate="one_to_one",
    )

    # 결측 라벨은 공식 평가에서도 해당 지표의 채점 대상에서 제외된다.
    evaluation["model_smape"] = smape_element(evaluation["pred_conc"], evaluation["target_conc_t2"])
    evaluation["baseline_smape"] = smape_element(
        evaluation["conc_3wk_avg"], evaluation["target_conc_t2"]
    )
    evaluation["smape_gain"] = evaluation["baseline_smape"] - evaluation["model_smape"]
    evaluation["alert_correct"] = evaluation["pred_alert"] == evaluation["target_alert_t2"]
    evaluation["origin_year"] = evaluation["origin_week"].str[:4].astype(int)
    evaluation["origin_week_number"] = evaluation["origin_week"].str[-2:].astype(int)
    evaluation["target_week_number"] = evaluation["date_week"].str[-2:].astype(int)
    return evaluation


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """추가 라이브러리 없이 작은 DataFrame을 Markdown 표로 변환한다."""

    selected = frame.loc[:, columns].copy()
    for column in selected.select_dtypes(include="number"):
        selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in selected.itertuples(index=False, name=None))
    return lines


def build_report(evaluation: pd.DataFrame, predictions_path: Path) -> str:
    """Q1 실패 원인과 다음 실험 우선순위를 Markdown 보고서로 만든다."""

    regression = evaluation.dropna(subset=["target_conc_t2", "conc_3wk_avg"]).copy()
    q1 = regression[regression["fold_id"].isin(["2025-Q1", "2026-Q1"])].copy()

    fold_summary = (
        regression.groupby("fold_id", as_index=False)
        .agg(
            rows=("region", "size"),
            model_smape=("model_smape", "mean"),
            baseline_smape=("baseline_smape", "mean"),
            mean_gain=("smape_gain", "mean"),
            missing_origin=("conc_mean", lambda values: values.isna().mean()),
        )
        .sort_values("fold_id")
    )
    region_summary = (
        q1.groupby("region", as_index=False)
        .agg(
            rows=("date_week", "size"),
            model_smape=("model_smape", "mean"),
            baseline_smape=("baseline_smape", "mean"),
            mean_gain=("smape_gain", "mean"),
            alert_accuracy=("alert_correct", "mean"),
        )
        .sort_values("mean_gain")
    )
    week_summary = (
        q1.groupby(["fold_id", "target_week_number"], as_index=False)
        .agg(
            rows=("region", "size"),
            model_smape=("model_smape", "mean"),
            baseline_smape=("baseline_smape", "mean"),
            mean_gain=("smape_gain", "mean"),
        )
        .sort_values("mean_gain")
    )
    worst_rows = q1.nsmallest(15, "smape_gain").copy()

    # 로그 농도의 2주 실제 변화량이 큰 구간에서 성능이 무너지는지 확인한다.
    q1["actual_log_change"] = np.log1p(q1["target_conc_t2"]) - np.log1p(q1["conc_mean"])
    q1["absolute_log_change"] = q1["actual_log_change"].abs()
    q1["change_group"] = pd.qcut(
        q1["absolute_log_change"],
        q=[0.0, 0.5, 0.75, 0.9, 1.0],
        labels=["0~50%", "50~75%", "75~90%", "90~100%"],
        duplicates="drop",
    )
    change_summary = (
        q1.groupby("change_group", observed=True, as_index=False)
        .agg(
            rows=("region", "size"),
            model_smape=("model_smape", "mean"),
            baseline_smape=("baseline_smape", "mean"),
            mean_gain=("smape_gain", "mean"),
            mean_abs_log_change=("absolute_log_change", "mean"),
        )
    )

    lines = [
        "# Q1 오차 분석",
        "",
        f"예측 파일: `{predictions_path.relative_to(ROOT)}`",
        "",
        "## 전체 평가 창",
        "",
        *markdown_table(
            fold_summary,
            ["fold_id", "rows", "model_smape", "baseline_smape", "mean_gain", "missing_origin"],
        ),
        "",
        "`mean_gain`은 기준선 sMAPE에서 모델 sMAPE를 뺀 값이며 양수일수록 모델이 우수하다.",
        "",
        "## Q1 지역별 취약 구간",
        "",
        *markdown_table(
            region_summary.head(10),
            ["region", "rows", "model_smape", "baseline_smape", "mean_gain", "alert_accuracy"],
        ),
        "",
        "## Q1 주차별 최악 구간",
        "",
        *markdown_table(
            week_summary.head(12),
            ["fold_id", "target_week_number", "rows", "model_smape", "baseline_smape", "mean_gain"],
        ),
        "",
        "## 실제 2주 변화 크기별 성능",
        "",
        *markdown_table(
            change_summary,
            [
                "change_group",
                "rows",
                "mean_abs_log_change",
                "model_smape",
                "baseline_smape",
                "mean_gain",
            ],
        ),
        "",
        "## 기준선 대비 가장 나빠진 행",
        "",
        *markdown_table(
            worst_rows,
            [
                "fold_id",
                "region",
                "origin_week",
                "date_week",
                "conc_mean",
                "target_conc_t2",
                "pred_conc",
                "model_smape",
                "baseline_smape",
                "smape_gain",
            ],
        ),
        "",
        "## 다음 실험 판단 기준",
        "",
        "- 특정 지역·주차 상수를 추가하지 않고 반복되는 데이터 특성으로만 개선한다.",
        "- Q1 개선이 다른 8개 창의 점수를 훼손하지 않는지 공식 전체 평가로 확인한다.",
        "- 평균 점수뿐 아니라 최저 창 점수와 회귀 기준선 승리 창 수를 함께 기록한다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="공식 백테스트 예측 오차 분석")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(f"예측 파일이 없습니다: {predictions_path}")

    evaluation = load_evaluation_rows(predictions_path)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(DETAIL, index=False, encoding="utf-8-sig")
    REPORT.write_text(build_report(evaluation, predictions_path), encoding="utf-8")
    print(f"상세 오차: {DETAIL}")
    print(f"분석 보고서: {REPORT}")


if __name__ == "__main__":
    main()
