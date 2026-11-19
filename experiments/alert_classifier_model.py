"""직접 급증 경보 분류기를 비교하기 위한 실험 모델.

회귀 예측은 현재 ``src/model.py``를 그대로 사용하고, 경보 출력만
HistGradientBoostingClassifier의 확률로 교체한다. ``ALERT_THRESHOLD``는 로컬
공식 백테스트에서 임계값을 선별하기 위한 환경변수이며 최종 제출 코드에는
고정된 검증값만 반영한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# 공식 평가기는 모델 파일이 있는 experiments 폴더를 작업 위치로 사용한다.
# 공개 소스의 회귀 모델과 특성 생성기를 재사용하도록 src를 import 경로에 넣는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_builder import FeatureBuilder  # noqa: E402
from model import Model as RegressionModel  # noqa: E402


class Model(RegressionModel):
    """기존 회귀 결과에 직접 학습한 급증 경보를 결합한다."""

    def __init__(self):
        super().__init__()
        self.alert_classifier = make_pipeline(
            # 회귀 모델과 동일하게 학습 창의 중앙값과 결측 플래그만 사용한다.
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=2026,
            ),
        )

    def fit(self, rows, national):
        # 부모 클래스가 같은 누수 방지 특성으로 회귀 모델을 먼저 학습한다.
        super().fit(rows, national)
        x, _, y_cls = self.builder.training_arrays()

        # Macro-F1은 양성과 음성을 같은 비중으로 보므로, 표본 수가 적은 양성
        # 클래스가 학습에서 묻히지 않도록 역빈도 표본 가중치를 적용한다.
        counts = np.bincount(y_cls, minlength=2)
        class_weights = len(y_cls) / (2.0 * np.maximum(counts, 1))
        sample_weight = class_weights[y_cls]
        # 불균형 보정 자체가 확률을 과도하게 높이는지 확인할 수 있도록 로컬
        # 실험에서만 가중치 사용 여부를 바꾼다. 최종 모델에는 환경변수를 두지 않는다.
        if os.environ.get("ALERT_BALANCED", "1") == "1":
            self.alert_classifier.fit(
                x,
                y_cls,
                histgradientboostingclassifier__sample_weight=sample_weight,
            )
        else:
            self.alert_classifier.fit(x, y_cls)

    def predict(self, rows, origin_week, national):
        # 회귀 농도는 검증된 부모 모델의 결과를 그대로 유지한다.
        regression_output = super().predict(rows, origin_week, national)
        builder = FeatureBuilder(rows, national)
        x, regions = builder.prediction_array(origin_week)
        probability = self.alert_classifier.predict_proba(x)[:, 1]

        # 임계값은 모든 선택 평가 창에 동일하게 적용한다. 특정 fold·주차·지역을
        # 조건으로 두지 않아 공개 평가 창 맞춤 규칙을 피한다.
        threshold = float(os.environ.get("ALERT_THRESHOLD", "0.5"))
        return {
            region: (regression_output[region][0], int(probability[idx] >= threshold))
            for idx, region in enumerate(regions)
        }
