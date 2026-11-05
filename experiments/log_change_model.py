"""t→t+2 로그 농도 변화량을 직접 예측하는 실험 모델."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# 공식 평가기의 작업 폴더는 이 파일이 있는 experiments이므로 src를 명시한다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asos_features import aggregate_asos  # noqa: E402
from feature_builder import FeatureBuilder  # noqa: E402
from model import Model as LevelModel  # noqa: E402


class Model(LevelModel):
    """기존 수준 모델과 동일한 특성으로 로그 변화량을 별도 학습한다."""

    def __init__(self):
        super().__init__()
        self.change_regressor = make_pipeline(
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
        # 부모 수준 모델과 변화량 모델이 같은 누수 방지 행·ASOS 특성을 사용한다.
        super().fit(rows, national, aux=aux)
        x, y_change = self.builder.training_change_arrays()
        self.change_regressor.fit(x, y_change)

    def predict(self, rows, origin_week, national, aux=None):
        # 이 실험 출력은 변화량 모델 단독 농도다. 이후 선별 스크립트가 기존
        # 수준 모델과 여러 비율로 섞어 재학습 없이 최적 후보를 찾는다.
        weather_weekly = aggregate_asos(aux)
        builder = FeatureBuilder(rows, national, weather_weekly)
        x, regions = builder.prediction_array(origin_week)
        predicted_change = self.change_regressor.predict(x)
        current = np.asarray(
            [builder.reference_concentration(region, origin_week) for region in regions]
        )
        pred_conc = np.maximum(0.0, np.expm1(np.log1p(current) + predicted_change))

        # 단독 모델도 공식 반환 형식을 만족하도록 기존과 같은 회귀 연계 경보를 계산한다.
        current_rows = {row["region"]: row for row in rows if row["date_week"] == origin_week}
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
