"""학습 모델 전에 빠르게 비교하는 규칙형 실험 모델.

SIGNAL_RULE·SIGNAL_SCALE·SIGNAL_ALERT_MULT 환경변수는 로컬 실험에만
사용한다. 최종 제출 모델은 ``src/model.py``이며 환경변수에 의존하지 않는다.
"""

import math
import os


def number(value):
    """평가기의 빈 문자열을 규칙 계산에서 쓰는 None으로 바꾼다."""
    return None if value in (None, "") else float(value)


def geometric(values):
    """농도처럼 배율로 변하는 값에 적합한 log1p 공간 평균을 계산한다."""
    values = [max(0.0, value) for value in values if value is not None]
    return math.expm1(sum(math.log1p(value) for value in values) / len(values)) if values else 0.0


class Model:
    def fit(self, rows, national):
        # 과거값만 사용하는 규칙형 기준선이라 별도의 학습 단계가 없다.
        pass

    def predict(self, rows, origin_week, national):
        # 동일한 공식 평가 조건에서 여러 지속·평균·계절 규칙을 빠르게 비교한다.
        rule = os.environ.get("SIGNAL_RULE", "avg4")
        by_region = {}
        for row in rows:
            by_region.setdefault(row["region"], []).append(row)
        out = {}
        for region, history in by_region.items():
            history.sort(key=lambda row: row["date_week"])
            current = next(row for row in history if row["date_week"] == origin_week)
            concentrations = [number(row["conc_mean"]) for row in history]
            observed = [value for value in concentrations if value is not None]
            current_conc = number(current["conc_mean"])
            three_week = number(current["conc_3wk_avg"])
            # current: 현재값 지속, avgN: 최근 N개 유효 관측의 기하평균,
            # seasonal: 약 1년 전 값, trend: 최근 변화율 연장, blend: 혼합 규칙이다.
            if rule == "current":
                pred = current_conc if current_conc is not None else observed[-1]
            elif rule == "avg2":
                pred = geometric(observed[-2:])
            elif rule == "avg4":
                pred = geometric(observed[-4:])
            elif rule == "avg8":
                pred = geometric(observed[-8:])
            elif rule == "seasonal":
                pred = concentrations[-53] if len(concentrations) >= 53 and concentrations[-53] is not None else geometric(observed[-4:])
            elif rule == "trend":
                recent = geometric(observed[-2:])
                prior = geometric(observed[-4:-2])
                pred = recent * (recent / prior) if prior > 0 else recent
            elif rule == "blend":
                pred = geometric([three_week, geometric(observed[-4:]), current_conc])
            else:
                pred = three_week if three_week is not None else observed[-1]
            # 회귀 보정계수와 경보 임계값을 분리해 각 점수 성분을 탐색한다.
            # 특정 fold 이름이나 주차를 조건으로 사용하지 않는다.
            pred *= float(os.environ.get("SIGNAL_SCALE", "1.0"))
            base = number(current["conc_base_avg"])
            alert_multiplier = float(os.environ.get("SIGNAL_ALERT_MULT", "1.3"))
            alert = int(base is not None and pred >= alert_multiplier * base)
            out[region] = (max(0.0, pred), alert)
        return out
