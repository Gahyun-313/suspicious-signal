"""공식 ``aux`` ASOS 일자료를 시·도-주차 특성으로 집계한다."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _read_stations() -> list[dict]:
    """공식 재실행 환경이 제공하는 허용 정적 관측소 목록을 읽는다."""

    static_dir = os.environ.get("KOWAS_STATIC_DIR")
    if not static_dir:
        raise RuntimeError("KOWAS_STATIC_DIR 환경변수가 없어 ASOS 관측소를 읽을 수 없습니다")
    path = Path(static_dir) / "asos_stations.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_asos(aux=None) -> dict[tuple[str, str], dict]:
    """원점 이하 ASOS 자료만 지역·주차 단위로 집계한다.

    배포 집계 함수의 ``quality_separate``·``zero_assumption`` 정책을 따른다.
    강수와 평균기온의 완전성을 따로 판단하고 품질주의 행이 포함된
    관측소-주는 해당 변수 집계에서 제외한다.
    """

    aux = aux or {}
    daily = aux.get("weather_asos_daily", [])
    quality = aux.get("weather_quality_flags_v0.1", [])
    if not daily:
        return {}

    stations = _read_stations()
    station_region = {row["stn_id"]: row["region"] for row in stations}
    flags = {(row["stn_id"], row["date"]): row["quality_flag"] for row in quality}

    # 먼저 관측소·주차 단위로 일자료를 모아 7일 완전성을 검사한다.
    station_weeks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen = set()
    for row in daily:
        key = (row["stn_id"], row["date"])
        if key in seen:
            raise ValueError(f"ASOS 관측소·날짜 중복: {key}")
        seen.add(key)
        if row["stn_id"] not in station_region:
            raise ValueError(f"관측소 목록에 없는 ASOS ID: {row['stn_id']}")
        station_weeks[(row["stn_id"], row["date_week"])].append(row)

    regional = defaultdict(lambda: {"rain": [], "temp": [], "blank_rain_days": 0})
    for (station, week), rows in station_weeks.items():
        region = station_region[station]
        target = regional[(region, week)]

        # 공식 zero_assumption 정책에 따라 빈 강수는 0 mm로 계산하되,
        # 품질주의표에 강수 이상이 기록된 관측소-주는 통째로 제외한다.
        rain_valid = len(rows) == 7 and not any(
            "blank_precip_positive_duration" in flags.get((station, row["date"]), "")
            for row in rows
        )
        temp_valid = (
            len(rows) == 7
            and all(row["temp_avg"] != "" for row in rows)
            and not any(
                "temp_mean_outside" in flags.get((station, row["date"]), "") for row in rows
            )
        )
        if rain_valid:
            target["rain"].append(sum(float(row["precip_mm"] or 0.0) for row in rows))
            target["blank_rain_days"] += sum(row["precip_mm"] == "" for row in rows)
        if temp_valid:
            target["temp"].append(mean(float(row["temp_avg"]) for row in rows))

    # 관측소별 주간값을 시·도 안에서 단순평균해 패널의 공간 단위와 맞춘다.
    result = {}
    for (region, week), values in regional.items():
        result[(region, week)] = {
            "asos_precip": mean(values["rain"]) if values["rain"] else None,
            "asos_temp": mean(values["temp"]) if values["temp"] else None,
            "asos_n_precip": len(values["rain"]),
            "asos_n_temp": len(values["temp"]),
            "asos_blank_rain_days": values["blank_rain_days"],
        }
    return result
