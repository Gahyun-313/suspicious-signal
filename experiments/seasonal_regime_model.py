"""1분기의 변화량 모델 혼합비만 조정하는 계절 국면 실험 모델."""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import Model as BaselineModel  # noqa: E402


class Model(BaselineModel):
    """연초 분포 변화에 한해 별도 변화량 혼합비를 적용한다."""

    def change_blend_weight(self, origin_week):
        """1분기에는 실험 비율을, 나머지 기간에는 기존 10%를 반환한다."""

        # 평가는 목표 주보다 2주 앞선 원점을 전달하므로, 원점에 2주를 더해
        # 실제 예측 대상 주차를 구한다. 그래야 연말·분기 경계를 잘못 나누지 않는다.
        year_text, week_text = origin_week.split("-W")
        origin_date = date.fromisocalendar(int(year_text), int(week_text), 1)
        target_date = origin_date + timedelta(weeks=2)
        target_week = target_date.isocalendar().week

        # 목표 ISO 주차 1~13에만 실험 비율을 적용한다. 후보 비율은 환경변수로
        # 바꿔 같은 코드와 평가 조건에서 비교하고, 연도별 상수는 두지 않는다.
        if target_week <= 13:
            return float(os.environ.get("Q1_CHANGE_WEIGHT", "0.1"))
        return super().change_blend_weight(origin_week)
