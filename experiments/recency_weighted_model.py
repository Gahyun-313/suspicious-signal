"""최근 관측에 더 높은 표본 가중치를 주는 HGB 실험 모델."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asos_features import aggregate_asos  # noqa: E402
from feature_builder import FeatureBuilder  # noqa: E402
from model import Model as BaselineModel  # noqa: E402


def recency_weights(builder, half_life: float, require_alert: bool) -> np.ndarray:
    """학습 창 최신 주를 기준으로 지수 감소 표본 가중치를 만든다."""

    weeks = sorted({row["date_week"] for rows in builder.rows_by_region.values() for row in rows})
    week_index = {week: index for index, week in enumerate(weeks)}
    latest_index = len(weeks) - 1
    weights = []

    # training_arrays와 같은 지역·주차 순서로 순회해야 각 가중치가 같은 행에 대응한다.
    for region in builder.region_order:
        for row in builder.rows_by_region[region]:
            if row.get("target_conc_t2", "") == "":
                continue
            if require_alert and row.get("target_alert_t2", "") == "":
                continue
            age = latest_index - week_index[row["date_week"]]
            weights.append(0.5 ** (age / half_life))

    weights = np.asarray(weights, dtype=float)
    # 평균을 1로 맞춰 기존 규제 강도의 의미가 가중치 크기 때문에 달라지지 않게 한다.
    return weights / weights.mean()


class Model(BaselineModel):
    """예측·앙상블은 유지하고 두 회귀기의 학습 표본 가중치만 바꾼다."""

    def fit(self, rows, national, aux=None):
        weather_weekly = aggregate_asos(aux)
        self.builder = FeatureBuilder(rows, national, weather_weekly)
        half_life = float(os.environ.get("RECENCY_HALF_LIFE", "52"))

        x, y_reg, _ = self.builder.training_arrays()
        level_weights = recency_weights(self.builder, half_life, require_alert=True)
        self.regressor.fit(
            x,
            y_reg,
            histgradientboostingregressor__sample_weight=level_weights,
        )

        change_x, y_change = self.builder.training_change_arrays()
        change_weights = recency_weights(self.builder, half_life, require_alert=False)
        self.change_regressor.fit(
            change_x,
            y_change,
            histgradientboostingregressor__sample_weight=change_weights,
        )
