"""공식 v2.5 데이터의 재현 가능한 Markdown 프로파일을 생성한다."""

from pathlib import Path

import pandas as pd


# 실행 위치와 무관하게 프로젝트 루트를 기준으로 원본과 결과 경로를 찾는다.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "KOWAS-EPI_전체데이터_v2.5" / "KOWAS-EPI"
OUT = ROOT / "docs" / "data-profile.md"


def main():
    # region_code는 01~17 형태이므로 정수로 읽어 앞자리 0이 사라지지 않게 한다.
    panel = pd.read_csv(DATA / "kowas_epi_panel.csv", dtype={"region_code": str})
    folds = pd.read_csv(DATA / "folds.csv")
    national = pd.read_csv(DATA / "national_weekly.csv")
    plants = pd.read_csv(DATA / "plant_meta.csv", dtype={"region_code": str})

    # 기본 규모, 클래스 불균형, 결측, 키 중복을 한 번에 다시 확인할 수 있게 집계한다.
    alert = panel["target_alert_t2"].dropna()
    missing = panel.isna().sum().sort_values(ascending=False)
    duplicate_keys = panel.duplicated(["date_week", "region"]).sum()
    # 분석 결과를 사람이 검토하고 Git으로 변경 이력을 남기기 쉬운 Markdown으로 만든다.
    lines = [
        "# KOWAS-EPI v2.5 데이터 프로파일",
        "",
        "이 문서는 `scripts/analyze_data.py`로 생성했습니다.",
        "",
        "## 구성",
        "",
        f"- 패널: {len(panel):,}행 × {panel.shape[1]}열, {panel['region'].nunique()}개 시·도, "
        f"{panel['date_week'].nunique()}주 ({panel['date_week'].min()} ~ {panel['date_week'].max()})",
        f"- 전국 주간 자료: {len(national):,}행 × {national.shape[1]}열",
        f"- 처리장 메타: {len(plants):,}행, {plants['region'].nunique()}개 시·도",
        f"- 공식 평가: {len(folds)}개 롤링 창, 회귀 {folds['n_regression_labels'].sum():,}건, "
        f"분류 {folds['n_alert_labels'].sum():,}건",
        f"- 패널 키 중복: {duplicate_keys}건",
        "",
        "## 타깃",
        "",
        "- `target_conc_t2`: 원점으로부터 2주 뒤 시·도별 하수 바이러스 농도",
        "- `target_alert_t2`: 목표 농도가 원점의 `conc_base_avg` 1.3배 이상인지 여부",
        f"- 분류 라벨 양성률: {alert.mean():.1%} ({int(alert.sum()):,}/{len(alert):,})",
        "",
        "## 주요 결측",
        "",
        "| 열 | 결측 수 | 결측률 |",
        "|---|---:|---:|",
    ]
    # 결측이 실제로 존재하는 열만 결측 수가 많은 순서로 기록한다.
    for column, count in missing.items():
        if count:
            lines.append(f"| `{column}` | {count:,} | {count / len(panel):.1%} |")

    lines.extend(
        [
            "",
            "## 수치 범위와 이상 신호",
            "",
            f"- 농도 범위: {panel['conc_mean'].min():,.2f} ~ {panel['conc_mean'].max():,.2f}; "
            "원 스케일 편차가 매우 커 로그 변환이 필요합니다.",
            f"- 전주 대비 증감률 범위: {panel['wow_change_rate'].min():,.2f} ~ "
            f"{panel['wow_change_rate'].max():,.2f}; 기간별 부호 규약도 달라 원값을 그대로 쓰지 않습니다.",
            f"- `pop_sampled > pop_served`: "
            f"{int((panel['pop_sampled'] > panel['pop_served']).sum()):,}행; 중복 집계 가능성이 있어 로그 변환 후 보조 변수로만 사용합니다.",
            "- v2.5 문서가 밝힌 대구 2025-W04 `conc_3wk_avg` 1행은 t+1 정보를 포함하므로 모델 입력에서 재계산 또는 제외가 필요합니다.",
            "- `plant_meta.in_current_registry`와 `last_sample_week`는 미래 시점 정보이므로 모델 입력에 사용하지 않습니다.",
            "",
            "## 검증 원칙",
            "",
            "공식 `evaluation.py`의 10개 롤링 창을 그대로 사용합니다. 각 창은 새 프로세스에서 학습하며 "
            "원점 이후 패널·전국·ASOS 자료와 아직 공개되지 않은 라벨을 가립니다. 모델 선택은 전체 창 평균뿐 아니라 "
            "최저 창 점수, 회귀 sMAPE 개선의 일관성, Macro-F1 균형을 함께 확인합니다.",
            "",
        ]
    )
    # docs 폴더가 없는 최초 실행도 지원하고 UTF-8로 한글을 보존한다.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
