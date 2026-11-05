"""KOWAS-EPI 공식 평가기에서 실행되는 누수 방지 베이스라인 모델."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from asos_features import aggregate_asos
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
        # 수준 모델이 놓치는 상승·하락 배율을 보완하도록 같은 특성으로
        # log(t+2 농도) - log(현재 농도)를 예측하는 두 번째 모델을 둔다.
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
        self.builder = None

    def fit(self, rows, national, aux=None):
        # rows와 national은 공식 평가기가 학습 기준 주까지만 잘라 전달한다.
        # 원본 CSV를 직접 읽지 않아 평가 창 이후 정보가 섞이지 않는다.
        # aux에는 평가기가 학습 기준 주까지 잘라 준 ASOS 일자료만 들어 있다.
        weather_weekly = aggregate_asos(aux)
        self.builder = FeatureBuilder(rows, national, weather_weekly)
        x, y_reg, y_cls = self.builder.training_arrays()
        # 현재 모델은 회귀 예측에서 경보를 유도하므로 y_cls는 사용하지 않는다.
        # 반환 계약을 유지해 이후 직접 분류 모델을 붙일 때 같은 특성을 쓸 수 있다.
        self.regressor.fit(x, y_reg)
        change_x, y_change = self.builder.training_change_arrays()
        self.change_regressor.fit(change_x, y_change)

    def predict(self, rows, origin_week, national, aux=None):
        # predict의 rows는 fit 때보다 뒤인 각 예측 원점까지 포함한다.
        # 따라서 학습 때 저장한 builder를 재사용하지 않고 원점 기준으로 다시 만든다.
        # predict에서도 원점 주까지의 aux만 새로 집계해 미래 기상 누수를 막는다.
        weather_weekly = aggregate_asos(aux)
        builder = FeatureBuilder(rows, national, weather_weekly)
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
        # 학습 모델 50%, 현재 농도 50%를 로그 공간에서 섞어 분포 변화 위험을 줄인다.
        alpha = 0.5
        blended_log = alpha * learned_log + (1.0 - alpha) * np.log1p(current)

        # 배포본 conc_3wk_avg에는 미래값이 섞인 것으로 알려진 행이 하나 있다.
        # 따라서 그 열을 출력 블렌딩에 직접 쓰지 않고, 평가기가 전달한 원점 이하
        # conc_mean만으로 지역별 최근 3개 유효 관측의 산술평균을 다시 계산한다.
        safe_three_week = []
        for region in regions:
            history = [
                float(row["conc_mean"])
                for row in rows
                if row["region"] == region
                and row["date_week"] <= origin_week
                and row["conc_mean"] != ""
            ][-3:]
            safe_three_week.append(float(np.mean(history)) if history else np.nan)
        safe_three_week = np.asarray(safe_three_week)

        # 안전한 후행 평균 30%를 추가하면 급변 구간의 과도한 현재값 추종을 줄이면서
        # 10개 창 평균과 최저 창 점수가 함께 개선됐다. 후행 평균이 없으면 기존
        # 블렌딩 예측을 그대로 사용한다.
        safe_log = np.where(np.isfinite(safe_three_week), np.log1p(safe_three_week), blended_log)
        blended_log = 0.7 * blended_log + 0.3 * safe_log
        # log1p를 원 단위(copies/mL)로 되돌리고 물리적으로 불가능한 음수를 막는다.
        level_conc = np.maximum(0.0, np.expm1(blended_log))

        # 변화량 모델은 현재 농도에 예측 로그 변화를 더해 t+2 농도로 복원한다.
        # 단독 성능은 낮지만 급변 구간에서 보완 효과가 있어 전체 롤링 창에서
        # 검증한 전역 비율 10%만 수준 모델에 섞는다.
        predicted_change = self.change_regressor.predict(x)
        change_conc = np.maximum(0.0, np.expm1(np.log1p(current) + predicted_change))
        final_log = 0.9 * np.log1p(level_conc) + 0.1 * np.log1p(change_conc)
        pred_conc = np.maximum(0.0, np.expm1(final_log))

        # 경보는 최종 t+2 농도 예측 90%와 현재 농도 10%를 기준선 대비
        # 로그 비율로 결합한다. 회귀 예측과 경보 판단이 같은 미래 신호를
        # 공유하면서도 현재 관측의 지속성을 조금 남기는 구조다.
        pred_alert = []
        for idx, region in enumerate(regions):
            base_text = current_rows[region]["conc_base_avg"]
            if base_text == "":
                pred_alert.append(0)
                continue
            base = max(float(base_text), 1e-9)
            predicted_ratio = np.log((pred_conc[idx] + 1.0) / (base + 1.0))
            current_ratio = np.log((current[idx] + 1.0) / (base + 1.0))
            combined_ratio = 0.9 * predicted_ratio + 0.1 * current_ratio
            # 임계값 1.1은 특정 창이 아니라 공개된 10개 롤링 창 전체에
            # 하나의 값으로 적용해 Macro-F1과 최저 창 점수를 함께 비교했다.
            pred_alert.append(int(combined_ratio >= np.log(1.1)))
        pred_alert = np.asarray(pred_alert)
        # 공식 평가기가 요구하는 {지역: (예측 농도, 0/1 경보)} 형식으로 반환한다.
        return {
            region: (float(conc), int(alert))
            for region, conc, alert in zip(regions, pred_conc, pred_alert)
        }
