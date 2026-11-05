"""KOWAS-EPI 공식 평가기에서 실행되는 누수 방지 베이스라인 모델."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from feature_builder import FeatureBuilder


class Model:
    """2주 뒤 농도를 예측하고 현재 농도로 급증 경보를 판단한다."""

    def __init__(self):
        # 농도는 지역·시점별 편차가 매우 크므로 원 농도가 아니라
        # feature_builder에서 만든 log1p 농도를 학습한다.
        self.regressor = make_pipeline(
            # 공식 평가기가 결측을 빈 문자열로 전달한다. 특성 생성 단계에서는
            # 이를 NaN으로 보존하고 여기서 학습 창의 중앙값으로만 대체한다.
            # add_indicator=True는 값이 없었다는 사실도 별도 특성으로 남긴다.
            SimpleImputer(strategy="median", add_indicator=True),
            # 표본이 약 3천 행으로 작기 때문에 복잡한 딥러닝보다 규제된
            # 히스토그램 부스팅을 첫 학습형 기준선으로 사용한다.
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=2026,
            ),
        )
        self.builder = None

    def fit(self, rows, national):
        # rows와 national은 공식 평가기가 학습 기준 주까지만 잘라 전달한다.
        # 원본 CSV를 직접 읽지 않아 평가 창 이후 정보가 섞이지 않는다.
        self.builder = FeatureBuilder(rows, national)
        x, y_reg, y_cls = self.builder.training_arrays()
        # 현재 모델은 회귀 예측에서 경보를 유도하므로 y_cls는 사용하지 않는다.
        # 반환 계약을 유지해 이후 직접 분류 모델을 붙일 때 같은 특성을 쓸 수 있다.
        self.regressor.fit(x, y_reg)

    def predict(self, rows, origin_week, national):
        # predict의 rows는 fit 때보다 뒤인 각 예측 원점까지 포함한다.
        # 따라서 학습 때 저장한 builder를 재사용하지 않고 원점 기준으로 다시 만든다.
        builder = FeatureBuilder(rows, national)
        x, regions = builder.prediction_array(origin_week)
        learned_log = self.regressor.predict(x)

        # 원점 주의 농도를 지역별로 찾는다. 원점 값이 결측이면 0으로 두지 않고
        # 그 원점 이전의 가장 최근 관측값을 사용해 급격한 과소예측을 막는다.
        current_rows = {row["region"]: row for row in rows if row["date_week"] == origin_week}
        latest = {}
        for row in rows:
            if row["date_week"] <= origin_week and row["conc_mean"] != "":
                latest[row["region"]] = float(row["conc_mean"])
        current = np.asarray(
            [
                max(0.0, float(current_rows[region]["conc_mean"] or latest.get(region, 0.0)))
                for region in regions
            ]
        )
        # 10개 공개 롤링 창 전체에서 하나의 값으로 선택한 혼합비다.
        # 창별 상수를 두지 않아 특정 평가 기간 맞춤 규칙을 금지한 대회 규정을 지킨다.
        # 학습 모델 40%, 현재 농도 60%를 로그 공간에서 섞어 분포 변화 위험을 줄인다.
        alpha = 0.4
        blended_log = alpha * learned_log + (1.0 - alpha) * np.log1p(current)
        # log1p를 원 단위(copies/mL)로 되돌리고 물리적으로 불가능한 음수를 막는다.
        pred_conc = np.maximum(0.0, np.expm1(blended_log))

        # 공식 라벨은 t+2 농도가 원점 기준선의 1.3배 이상인지로 정의되지만,
        # 현재 농도에 적용할 지속 경보 임계값 1.1은 전체 창에서 공통으로 선택했다.
        pred_alert = np.asarray(
            [
                int(
                    current_rows[region]["conc_base_avg"] != ""
                    and current[idx] >= 1.1 * float(current_rows[region]["conc_base_avg"])
                )
                for idx, region in enumerate(regions)
            ]
        )
        # 공식 평가기가 요구하는 {지역: (예측 농도, 0/1 경보)} 형식으로 반환한다.
        return {
            region: (float(conc), int(alert))
            for region, conc, alert in zip(regions, pred_conc, pred_alert)
        }
