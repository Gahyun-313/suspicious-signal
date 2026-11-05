"""수준·변화량 HGB의 전역 복잡도 설정을 비교하는 실험 모델."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# 공식 평가기는 이 파일의 폴더를 작업 위치로 사용하므로 src를 명시한다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import Model as BaselineModel  # noqa: E402


def build_regressor(max_leaf_nodes: int, min_samples_leaf: int, l2: float):
    """수준·변화량 모델에 공통으로 적용할 HGB 파이프라인을 만든다."""

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=250,
            # 리프 수가 많을수록 복잡한 상호작용을 표현하지만 과적합 위험도 커진다.
            max_leaf_nodes=max_leaf_nodes,
            # 리프 최소 표본 수는 작은 지역·주차 패턴을 얼마나 허용할지 제어한다.
            min_samples_leaf=min_samples_leaf,
            # L2 규제는 각 리프 예측값의 과도한 변동을 억제한다.
            l2_regularization=l2,
            random_state=2026,
        ),
    )


class Model(BaselineModel):
    """현재 모델의 데이터·특성·앙상블은 유지하고 HGB 복잡도만 바꾼다."""

    def __init__(self):
        super().__init__()
        # 환경변수는 로컬 실험 선별에만 사용한다. 채택 설정은 최종 모델 코드에
        # 상수로 고정해 공식 재실행 결과가 환경에 따라 달라지지 않게 한다.
        leaves = int(os.environ.get("HGB_LEAVES", "15"))
        minimum = int(os.environ.get("HGB_MIN_SAMPLES", "20"))
        l2 = float(os.environ.get("HGB_L2", "2.0"))
        self.regressor = build_regressor(leaves, minimum, l2)
        self.change_regressor = build_regressor(leaves, minimum, l2)
