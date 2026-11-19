"""저장된 회귀 예측으로 경보 결합비와 임계값을 빠르게 재채점한다."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


def shift_week(week_text: str, offset: int) -> str:
    """ISO 주차 문자열을 연말 경계를 포함해 안전하게 이동한다."""

    year_text, number_text = week_text.split("-W")
    shifted = date.fromisocalendar(int(year_text), int(number_text), 1) + timedelta(weeks=offset)
    iso_year, iso_week, _ = shifted.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def macro_f1(pairs: list[tuple[int, int]]) -> float:
    """공식 평가기와 같은 이진 Macro-F1을 계산한다."""

    scores = []
    for label in (0, 1):
        tp = sum(pred == label and true == label for pred, true in pairs)
        fp = sum(pred == label and true != label for pred, true in pairs)
        fn = sum(pred != label and true == label for pred, true in pairs)
        scores.append(1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / 2


def load_csv(path: Path) -> list[dict[str, str]]:
    """UTF-8 CSV를 문자열 사전 목록으로 읽는다."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_context(panel: list[dict[str, str]]) -> dict[tuple[str, str], tuple[float, str]]:
    """각 지역·원점에서 모델이 사용한 현재 농도와 기준선을 복원한다."""

    by_region = defaultdict(list)
    for row in panel:
        by_region[row["region"]].append(row)

    context = {}
    for region, rows in by_region.items():
        latest = 0.0
        for row in sorted(rows, key=lambda item: item["date_week"]):
            if row["conc_mean"] != "":
                latest = max(0.0, float(row["conc_mean"]))
            # 원점 농도 결측 시 model.py와 동일하게 최근 유효값을 사용한다.
            context[(region, row["date_week"])] = (latest, row["conc_base_avg"])
    return context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-json", type=Path, required=True)
    args = parser.parse_args()

    panel = load_csv(args.data_dir / "kowas_epi_panel.csv")
    predictions = load_csv(args.predictions)
    context = build_context(panel)
    official = json.loads(args.score_json.read_text(encoding="utf-8"))
    score_parts = {row["fold_id"]: row for row in official["folds"]}

    # 목표 주의 정답은 원점 행에 target_*로 붙어 있으므로 키를 목표 주로 바꾼다.
    truths = {}
    for row in panel:
        if row["target_alert_t2"] != "":
            truths[(row["region"], row["target_week_t2"])] = int(row["target_alert_t2"])

    folds = load_csv(args.data_dir / "folds.csv")
    candidates = []
    # 넓은 후보와 현재 최선 주변의 촘촘한 후보를 한 번에 비교한다.
    predicted_weights = (0.7, 0.8, 0.85, 0.9, 0.95, 1.0)
    thresholds = (1.0, 1.025, 1.05, 1.075, 1.08, 1.09, 1.1, 1.11, 1.12, 1.125, 1.15, 1.2)
    for predicted_weight in predicted_weights:
        for threshold in thresholds:
            pairs_by_fold = defaultdict(list)
            for prediction in predictions:
                key = (prediction["region"], prediction["date_week"])
                if key not in truths:
                    continue
                origin_week = shift_week(prediction["date_week"], -2)
                current, base_text = context[(prediction["region"], origin_week)]
                if base_text == "":
                    alert = 0
                else:
                    base = max(float(base_text), 1e-9)
                    predicted_ratio = math.log((float(prediction["pred_conc"]) + 1.0) / (base + 1.0))
                    current_ratio = math.log((current + 1.0) / (base + 1.0))
                    ratio = predicted_weight * predicted_ratio + (1.0 - predicted_weight) * current_ratio
                    alert = int(ratio >= math.log(threshold))

                # 목표 주가 속한 공식 창을 찾아 창별 Macro-F1을 따로 계산한다.
                target_week = prediction["date_week"]
                fold_id = next(
                    fold["fold_id"]
                    for fold in folds
                    if fold["target_start"] <= target_week <= fold["target_end"]
                )
                pairs_by_fold[fold_id].append((alert, truths[key]))

            fold_scores = []
            for fold in folds:
                fold_id = fold["fold_id"]
                f1 = macro_f1(pairs_by_fold[fold_id])
                fixed = score_parts[fold_id]
                score = 0.6 * (1.0 - fixed["smape"] / fixed["smape_baseline"]) + 0.4 * f1
                fold_scores.append(score)
            candidates.append(
                {
                    "predicted_weight": predicted_weight,
                    "threshold": threshold,
                    "score": sum(fold_scores) / len(fold_scores),
                    "worst_fold": min(fold_scores),
                }
            )

    # 평균 점수를 우선하고 동률이면 최저 창이 높은 후보를 앞에 둔다.
    candidates.sort(key=lambda row: (row["score"], row["worst_fold"]), reverse=True)
    for row in candidates[:12]:
        print(
            f"predicted_weight={row['predicted_weight']:.2f} "
            f"threshold={row['threshold']:.3f} score={row['score']:.6f} "
            f"worst={row['worst_fold']:.6f}"
        )


if __name__ == "__main__":
    main()
