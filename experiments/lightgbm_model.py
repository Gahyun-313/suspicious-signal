"""현재 특성·앙상블을 유지한 LightGBM 회귀 모델 비교본."""

from __future__ import annotations

import sys
from pathlib import Path

from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# 공식 평가기의 작업 폴더는 experiments이므로 공개 모델이 있는 src를 추가한다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import Model as HGBModel  # noqa: E402


def build_regressor():
    """작은 주간 패널에 과도하게 적합되지 않도록 규제한 LightGBM을 만든다."""

    return make_pipeline(
        # LightGBM은 NaN을 처리할 수 있지만 결측 플래그를 명시적으로 남겨
        # 기존 HGB 모델과 동일한 입력 정보로 공정하게 비교한다.
        SimpleImputer(strategy="median", add_indicator=True),
        LGBMRegressor(
            objective="regression_l1",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            # 공식 평가기는 창별 프로세스를 이미 분리하므로 내부 병렬화는 끈다.
            n_jobs=1,
            random_state=2026,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ),
    )


class Model(HGBModel):
    """HGB와 데이터 처리·출력 규칙을 같게 두고 모델 계열만 바꾼다."""

    def __init__(self):
        super().__init__()
        self.regressor = build_regressor()
        self.change_regressor = build_regressor()
