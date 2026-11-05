"""문서자료와 모델자료 제출 ZIP을 재현 가능하게 생성한다."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]


def register_korean_fonts() -> tuple[str, str]:
    """Windows의 맑은 고딕을 등록해 한글이 깨지지 않게 한다."""

    candidates = [
        (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgunbd.ttf")),
        (Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"), Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Korean", str(regular)))
            pdfmetrics.registerFont(TTFont("KoreanBold", str(bold)))
            return "Korean", "KoreanBold"
    raise RuntimeError("PDF 생성에 사용할 한글 글꼴을 찾지 못했습니다")


def add_page_number(canvas, document) -> None:
    """모든 페이지 아래에 문서명과 페이지 번호를 표시한다."""

    canvas.saveState()
    canvas.setFont("Korean", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 12 * mm, "수상한 시그널 - 문제해결 방법 설명서")
    canvas.drawRightString(192 * mm, 12 * mm, str(document.page))
    canvas.restoreState()


def build_pdf(markdown_path: Path, output_path: Path) -> None:
    """제출 보고서 Markdown을 읽어 단정한 A4 PDF로 변환한다."""

    regular, bold = register_korean_fonts()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyKorean",
        parent=styles["BodyText"],
        fontName=regular,
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5,
    )
    title = ParagraphStyle(
        "TitleKorean",
        parent=body,
        fontName=bold,
        fontSize=22,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F4C5C"),
        spaceAfter=14,
    )
    heading = ParagraphStyle(
        "HeadingKorean",
        parent=body,
        fontName=bold,
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#0F4C5C"),
        spaceBefore=10,
        spaceAfter=7,
    )
    bullet = ParagraphStyle("BulletKorean", parent=body, leftIndent=10, firstLineIndent=-7)

    story = []
    table_rows = []
    in_table = False
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if line.startswith("|---"):
            continue
        if line.startswith("|") and line.endswith("|"):
            in_table = True
            table_rows.append([cell.strip().replace("**", "") for cell in line.strip("|").split("|")])
            continue
        if in_table:
            table = Table(table_rows, colWidths=[115 * mm, 38 * mm])
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, 0), bold),
                        ("FONTNAME", (0, 1), (-1, -1), regular),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEEF2")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([table, Spacer(1, 4 * mm)])
            table_rows, in_table = [], False

        if not line:
            story.append(Spacer(1, 2 * mm))
        elif line.startswith("# "):
            story.extend([Spacer(1, 30 * mm), Paragraph(line[2:], title), Spacer(1, 6 * mm)])
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], heading))
        elif line.startswith("- "):
            story.append(Paragraph("• " + line[2:].replace("`", ""), bullet))
        elif line[0:2].isdigit() and line[2:4] == ". ":
            story.append(Paragraph(line.replace("`", ""), bullet))
        else:
            story.append(Paragraph(line.replace("**", "").replace("`", ""), body))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="수상한 시그널 - 문제해결 방법 설명서",
        author="수상한 알파카",
    )
    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def zip_directory(source: Path, destination: Path) -> None:
    """제출 파일을 하나의 최상위 폴더 아래에 모아 압축한다."""

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, Path(source.name) / path.relative_to(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    staging = ROOT / "tmp" / "submission"
    if staging.exists():
        shutil.rmtree(staging)
    output_dir.mkdir(parents=True, exist_ok=True)

    document_dir = staging / "document_submission"
    model_dir = staging / "model_submission"
    document_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    report_source = ROOT / "submission" / "document" / "solution_report.md"
    shutil.copy2(report_source, document_dir / "solution_report.md")
    build_pdf(report_source, document_dir / "solution_report.pdf")
    # 업로드 전 내용을 바로 열어볼 수 있도록 동일 PDF의 미리보기도 남긴다.
    shutil.copy2(document_dir / "solution_report.pdf", output_dir / "solution_report_preview.pdf")

    # 제출 모델에는 실행에 필요한 최소 파일만 포함해 원본 데이터 유출을 막는다.
    for source, name in (
        (ROOT / "src" / "model.py", "model.py"),
        (ROOT / "src" / "feature_builder.py", "feature_builder.py"),
        (ROOT / "src" / "asos_features.py", "asos_features.py"),
        (ROOT / "requirements.txt", "requirements.txt"),
        (ROOT / "submission" / "model" / "RUN_MANUAL.md", "RUN_MANUAL.md"),
    ):
        shutil.copy2(source, model_dir / name)

    zip_directory(document_dir, output_dir / "01_document_submission.zip")
    zip_directory(model_dir, output_dir / "02_model_submission.zip")
    print(output_dir / "01_document_submission.zip")
    print(output_dir / "02_model_submission.zip")


if __name__ == "__main__":
    main()
