"""KOWAS-EPI 롤링 평가용 누수 방지 특성 생성기."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


NUMERIC_COLUMNS = (
    # 원점 주에 알 수 있는 값만 선언한다. target_* 라벨은 절대 포함하지 않는다.
    "conc_mean",
    "conc_3wk_avg",
    "conc_base_avg",
    "n_sites",
    "precip_mm",
    "temp_avg",
    "pop_sampled",
)


def _number(value, default=np.nan):
    """평가기의 빈 문자열 결측을 NaN으로 통일한다."""
    try:
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _safe_log(value):
    """농도·인구처럼 치우친 양수 변수를 안전하게 log1p 변환한다."""
    return math.log1p(max(0.0, value)) if np.isfinite(value) else np.nan


def _week_parts(date_week):
    """대회 주차 문자열(예: 2026-W03)을 연도와 주 번호로 분리한다."""
    year, week = date_week.split("-W")
    return int(year), int(week)


class FeatureBuilder:
    """공식 평가기가 전달한 시점 이하 행만으로 특성을 만든다."""

    def __init__(self, rows, national, weather_weekly=None):
        # 지역별 과거 이력을 주차 순서로 정렬한다. 전달받은 rows 자체가
        # 기준 주 이후를 포함하지 않으므로 이 저장소에는 미래 관측이 없다.
        self.rows_by_region = defaultdict(list)
        for row in rows:
            self.rows_by_region[row["region"]].append(row)
        for region in self.rows_by_region:
            self.rows_by_region[region].sort(key=lambda row: row["date_week"])

        # 전국 자료도 평가기가 같은 기준 주로 잘라 전달한 값만 인덱싱한다.
        self.national = {row["date_week"]: row for row in national}
        # ASOS는 공식 aux를 집계한 선택 특성이다. 전달되지 않으면 기존 모델과
        # 같은 NaN 특성을 만들어 기상 미사용 환경에서도 인터페이스를 유지한다.
        self.weather_weekly = weather_weekly or {}
        # 정렬된 지역 순서를 고정해 학습·예측 배열의 열 의미와 반환 순서를 맞춘다.
        self.region_order = sorted(self.rows_by_region)
        self.region_index = {name: idx for idx, name in enumerate(self.region_order)}

    def _regional_history(self, region, date_week):
        """요청한 원점 주까지의 지역 이력만 반환한다."""
        return [row for row in self.rows_by_region[region] if row["date_week"] <= date_week]

    def reference_concentration(self, region, date_week):
        """원점 농도 또는 그 이전의 가장 최근 유효 농도를 반환한다."""

        # 원점 결측을 0으로 바꾸면 실제 급락으로 오해하므로, 공식 평가기가
        # 전달한 원점 이하 이력에서 가장 가까운 유효 관측을 찾는다.
        for row in reversed(self._regional_history(region, date_week)):
            if row.get("conc_mean", "") != "":
                return max(0.0, float(row["conc_mean"]))
        return 0.0

    def make_one(self, region, date_week):
        """지역 한 곳·원점 한 주에 대응하는 수치 특성 벡터를 만든다."""
        history = self._regional_history(region, date_week)
        current = history[-1]
        year, week = _week_parts(date_week)
        # 주차를 숫자 하나로 넣으면 연말과 연초가 멀어 보이므로 sin/cos로
        # 순환 계절성을 표현한다. year는 장기적인 수준 변화용이다.
        features = [
            float(self.region_index[region]),
            float(year - 2023),
            math.sin(2.0 * math.pi * week / 52.0),
            math.cos(2.0 * math.pi * week / 52.0),
        ]

        # 원점 값과 결측 여부를 함께 넣는다. 결측 자체가 측정 상황을 나타내는
        # 신호일 수 있어 단순 대체 전에 0/1 플래그를 보존한다.
        for column in NUMERIC_COLUMNS:
            value = _number(current.get(column, ""))
            if column.startswith("conc_") or column == "pop_sampled":
                value = _safe_log(value)
            features.extend([value, float(not np.isfinite(value))])

        # 현재(t)부터 t-8까지의 지역 농도 지연값으로 단기 변화 흐름을 표현한다.
        # 없는 주는 임의 보간하지 않고 NaN으로 두어 학습 파이프라인에 맡긴다.
        for lag in range(0, 9):
            row = history[-1 - lag] if len(history) > lag else None
            value = _number(row.get("conc_mean", "")) if row else np.nan
            features.append(_safe_log(value))

        # 최근 3·4·6·8·12주마다 수준, 변동성, 관측률을 계산한다.
        # 서로 다른 창 길이는 급격한 유행과 완만한 추세를 함께 포착하기 위함이다.
        finite = [
            _number(row.get("conc_mean", ""))
            for row in history[-12:]
            if np.isfinite(_number(row.get("conc_mean", "")))
        ]
        for window in (3, 4, 6, 8, 12):
            values = [
                _number(row.get("conc_mean", ""))
                for row in history[-window:]
                if np.isfinite(_number(row.get("conc_mean", "")))
            ]
            features.extend(
                [
                    _safe_log(float(np.mean(values))) if values else np.nan,
                    float(np.std([_safe_log(v) for v in values])) if len(values) > 1 else 0.0,
                    float(len(values)) / window,
                ]
            )
        # 가장 최근 두 유효 관측의 로그 차이는 배율 변화에 가까운 증감 신호다.
        # 부호 정의가 기간별로 바뀌는 원본 wow_change_rate 대신 직접 계산한다.
        if len(finite) >= 2:
            features.append(_safe_log(finite[-1]) - _safe_log(finite[-2]))
        else:
            features.append(np.nan)

        # 같은 원점 이하에서 확인 가능한 전국 농도·임상 신고수의 최근 5주를
        # 넣어 지역 변화와 전국 공통 유행을 구분할 단서를 제공한다.
        weeks = [row["date_week"] for row in history]
        for lag in range(0, 5):
            week_key = weeks[-1 - lag] if len(weeks) > lag else None
            national = self.national.get(week_key, {}) if week_key else {}
            features.extend(
                [
                    _safe_log(_number(national.get("conc_mean_national", ""))),
                    _safe_log(_number(national.get("covid_cases_national", ""))),
                ]
            )

        # 원점과 최근 4주 ASOS 관측 기상을 추가한다. 주차 키는 현재 지역
        # 이력에서 가져오므로 예측 원점 이후 기상은 조회할 수 없다.
        for lag in range(0, 5):
            week_key = weeks[-1 - lag] if len(weeks) > lag else None
            weather = self.weather_weekly.get((region, week_key), {}) if week_key else {}
            precipitation = _number(weather.get("asos_precip"))
            temperature = _number(weather.get("asos_temp"))
            features.extend(
                [
                    _safe_log(precipitation),
                    temperature,
                    _number(weather.get("asos_n_precip"), 0.0),
                    _number(weather.get("asos_n_temp"), 0.0),
                    _number(weather.get("asos_blank_rain_days"), 0.0),
                    float(not np.isfinite(precipitation)),
                    float(not np.isfinite(temperature)),
                ]
            )
        return features

    def training_arrays(self):
        """t+2 회귀·분류 라벨이 모두 공개된 행만 학습 배열로 변환한다."""
        x, y_reg, y_cls = [], [], []
        for region in self.region_order:
            for row in self.rows_by_region[region]:
                # 목표주가 학습 기준 주보다 뒤면 평가기가 라벨을 빈 문자열로
                # 가리므로 자동으로 제외된다. 별도 미래 주차 조회는 하지 않는다.
                if row.get("target_conc_t2", "") == "" or row.get("target_alert_t2", "") == "":
                    continue
                x.append(self.make_one(region, row["date_week"]))
                # 농도 타깃도 log1p로 학습해 극단적으로 큰 값의 영향력을 줄인다.
                y_reg.append(_safe_log(float(row["target_conc_t2"])))
                y_cls.append(int(row["target_alert_t2"]))
        return np.asarray(x, dtype=float), np.asarray(y_reg), np.asarray(y_cls)

    def training_change_arrays(self):
        """t에서 t+2까지의 로그 농도 변화량 학습 배열을 만든다."""

        x, y_change = [], []
        for region in self.region_order:
            for row in self.rows_by_region[region]:
                if row.get("target_conc_t2", "") == "":
                    continue
                current = self.reference_concentration(region, row["date_week"])
                target = max(0.0, float(row["target_conc_t2"]))
                x.append(self.make_one(region, row["date_week"]))
                # 로그 차이는 절대 농도보다 상승·하락의 배율 변화에 집중한다.
                y_change.append(_safe_log(target) - _safe_log(current))
        return np.asarray(x, dtype=float), np.asarray(y_change)

    def prediction_array(self, origin_week):
        """공식 평가 원점에서 17개 지역의 예측용 배열을 만든다."""
        x = [self.make_one(region, origin_week) for region in self.region_order]
        return np.asarray(x, dtype=float), self.region_order
