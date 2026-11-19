"""현재 특성·앙상블을 유지한 CatBoost 회귀 모델 비교본."""

from __future__ import annotations

import sys
from pathlib import Path

from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# 공식 평가기의 작업 폴더는 experiments이므로 공개 모델이 있는 src를 추가한다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import Model as HGBModel  # noqa: E402


def build_regressor():
    """작은 주간 패널에 맞춰 깊이와 규제를 제한한 CatBoost를 만든다."""

    return make_pipeline(
        # CatBoost도 결측을 처리하지만 기존 모델과 같은 결측 플래그를 제공해
        # 모델 계열 외의 차이가 결과에 섞이지 않도록 한다.
        SimpleImputer(strategy="median", add_indicator=True),
        CatBoostRegressor(
            loss_function="MAE",
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=2026,
            # 공식 평가기가 창별 프로세스를 분리하므로 모델 내부 병렬화는 끈다.
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    )


class Model(HGBModel):
    """HGB와 데이터 처리·출력 규칙을 같게 두고 모델 계열만 바꾼다."""

    def __init__(self):
        super().__init__()
        self.regressor = build_regressor()
        self.change_regressor = build_regressor()
