"""원점 전국 농도를 기준으로 지역별 t+2 편차를 예측하는 실험 모델."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asos_features import aggregate_asos  # noqa: E402
from feature_builder import FeatureBuilder  # noqa: E402


def national_concentration(builder, date_week):
    """원점 주 또는 그 이전의 가장 최근 전국 농도를 반환한다."""

    # 전국 자료도 평가기가 기준 주까지만 전달하므로 과거 방향으로 찾은 값에
    # 원점 이후 정보가 포함될 수 없다.
    available_weeks = sorted(week for week in builder.national if week <= date_week)
    for week in reversed(available_weeks):
        value = builder.national[week].get("conc_mean_national", "")
        if value != "":
            return max(0.0, float(value))
    return None


def training_arrays(builder):
    """원점 전국 농도를 기준으로 한 지역 t+2 편차 학습 배열을 만든다."""

    x, y_residual = [], []
    for region in builder.region_order:
        for row in builder.rows_by_region[region]:
            if row.get("target_conc_t2", "") == "":
                continue
            national_current = national_concentration(builder, row["date_week"])
            if national_current is None:
                continue
            target = max(0.0, float(row["target_conc_t2"]))
            x.append(builder.make_one(region, row["date_week"]))
            # 전국 절대 수준은 기준점으로 빼고 지역별 미래 편차만 학습한다.
            y_residual.append(np.log1p(target) - np.log1p(national_current))
    return np.asarray(x, dtype=float), np.asarray(y_residual)


class Model:
    """전국 공통 수준을 앵커로 두고 지역별 미래 편차만 학습한다."""

    def __init__(self):
        self.regressor = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=2026,
            ),
        )

    def fit(self, rows, national, aux=None):
        # 평가기가 학습 기준 주까지만 잘라 전달한 패널·전국·ASOS만 사용한다.
        weather_weekly = aggregate_asos(aux)
        builder = FeatureBuilder(rows, national, weather_weekly)
        x, y_residual = training_arrays(builder)
        self.regressor.fit(x, y_residual)

    def predict(self, rows, origin_week, national, aux=None):
        weather_weekly = aggregate_asos(aux)
        builder = FeatureBuilder(rows, national, weather_weekly)
        x, regions = builder.prediction_array(origin_week)
        predicted_residual = self.regressor.predict(x)

        # 예측한 지역 편차에 원점 전국 로그 농도를 다시 더해 지역 농도로 복원한다.
        national_current = national_concentration(builder, origin_week)
        if national_current is None:
            national_current = 0.0
        pred_conc = np.maximum(
            0.0,
            np.expm1(np.log1p(national_current) + predicted_residual),
        )

        # 단독 후보도 공식 형식을 만족하도록 현재 채택 모델과 같은 경보를 계산한다.
        current_rows = {row["region"]: row for row in rows if row["date_week"] == origin_week}
        current = np.asarray(
            [builder.reference_concentration(region, origin_week) for region in regions]
        )
        pred_alert = []
        for idx, region in enumerate(regions):
            base_text = current_rows[region]["conc_base_avg"]
            if base_text == "":
                pred_alert.append(0)
                continue
            base = max(float(base_text), 1e-9)
            predicted_ratio = np.log((pred_conc[idx] + 1.0) / (base + 1.0))
            current_ratio = np.log((current[idx] + 1.0) / (base + 1.0))
            pred_alert.append(int(0.9 * predicted_ratio + 0.1 * current_ratio >= np.log(1.1)))

        return {
            region: (float(pred_conc[idx]), int(pred_alert[idx]))
            for idx, region in enumerate(regions)
        }
