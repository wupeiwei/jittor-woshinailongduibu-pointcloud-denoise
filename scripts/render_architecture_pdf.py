#!/usr/bin/env python3
"""Render the project architecture Markdown into a print-friendly PDF.

This script intentionally avoids heavyweight HTML/PDF toolchains. It uses
Matplotlib's PDF backend directly, discovers a local Chinese font, and renders
the Markdown document as clean A4 pages with cover, navigation, headings,
paragraphs, lists, code blocks, tables, headers, and footers.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "PROJECT_ARCHITECTURE_CN.md"
DEFAULT_OUTPUT = ROOT / "docs" / "PROJECT_ARCHITECTURE_CN.pdf"


@dataclass
class Element:
    kind: str
    text: str = ""
    level: int = 0
    rows: list[list[str]] | None = None
    items: list[str] | None = None


def display_units(text: str) -> int:
    total = 0
    for ch in text:
        if ch == "\t":
            total += 4
        elif unicodedata.east_asian_width(ch) in {"W", "F", "A"}:
            total += 2
        else:
            total += 1
    return total


def strip_inline_markdown(text: str) -> str:
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def wrap_display(text: str, max_units: int) -> list[str]:
    text = strip_inline_markdown(text).strip()
    if not text:
        return [""]
    out: list[str] = []
    line = ""
    line_units = 0
    last_space = -1

    for ch in text:
        unit = display_units(ch)
        if line_units + unit > max_units and line:
            if last_space > 0 and len(line) - last_space < 18:
                out.append(line[:last_space].rstrip())
                line = line[last_space + 1 :].lstrip()
            else:
                out.append(line.rstrip())
                line = ""
            line_units = display_units(line)
            last_space = line.rfind(" ")
        line += ch
        line_units += unit
        if ch == " ":
            last_space = len(line) - 1

    if line.strip():
        out.append(line.strip())
    return out or [""]


def is_table_separator(line: str) -> bool:
    cells = [x.strip() for x in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)


def parse_table_line(line: str) -> list[str]:
    return [strip_inline_markdown(x.strip()) for x in line.strip().strip("|").split("|")]


def starts_block(line: str) -> bool:
    s = line.lstrip()
    return (
        s.startswith("#")
        or s.startswith("```")
        or s.startswith("- ")
        or bool(re.match(r"\d+\.\s+", s))
        or line.strip().startswith("|")
    )


def parse_markdown(path: Path) -> list[Element]:
    lines = path.read_text(encoding="utf-8").splitlines()
    elements: list[Element] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            fence = stripped[:3]
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(lines[i].rstrip())
                i += 1
            if i < len(lines):
                i += 1
            elements.append(Element("code", "\n".join(code)))
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            rows = [parse_table_line(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(parse_table_line(lines[i]))
                i += 1
            elements.append(Element("table", rows=rows))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            elements.append(Element("heading", strip_inline_markdown(heading.group(2)), level=len(heading.group(1))))
            i += 1
            continue

        if stripped.startswith("- ") or re.match(r"^\d+\.\s+", stripped):
            items: list[str] = []
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith("- "):
                    items.append(s[2:].strip())
                    i += 1
                elif re.match(r"^\d+\.\s+", s):
                    items.append(re.sub(r"^\d+\.\s+", "", s).strip())
                    i += 1
                elif not s:
                    i += 1
                    break
                else:
                    break
            elements.append(Element("list", items=items))
            continue

        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not starts_block(lines[i]):
            para.append(lines[i].strip())
            i += 1
        elements.append(Element("paragraph", " ".join(para)))
    return elements


def find_first_existing(paths: Iterable[str | Path]) -> Path | None:
    for path in paths:
        p = Path(path).expanduser()
        if p.exists():
            return p
    return None


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "缺少 matplotlib，无法生成 PDF。可在当前环境安装：python -m pip install matplotlib"
        ) from exc

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt, fm, PdfPages, Rectangle


class PdfRenderer:
    page_w = 595.28
    page_h = 841.89
    margin_l = 52
    margin_r = 46
    margin_t = 62
    margin_b = 58

    navy = "#16324f"
    blue = "#1f5f99"
    light_blue = "#edf4fb"
    ink = "#1f2933"
    muted = "#667085"
    grid = "#d0d7de"
    code_bg = "#f6f8fa"
    paper = "#ffffff"

    def __init__(self, pdf_path: Path, title: str, source_name: str, font_path: Path | None, bold_font_path: Path | None):
        self.plt, self.fm, self.PdfPages, self.Rectangle = setup_matplotlib()
        self.pdf = self.PdfPages(str(pdf_path))
        self.title = title
        self.source_name = source_name
        self.page_no = 0
        self.fig = None
        self.ax = None
        self.y = self.page_h - self.margin_t

        font_candidates = [
            font_path,
            os.environ.get("PDF_CJK_FONT", ""),
            ROOT / "assets" / "fonts" / "NotoSansSC-Regular.ttf",
            Path.home() / ".local/share/fonts/codex/NotoSansSC-Regular.ttf",
            Path.home() / ".local/share/fonts/codex/NotoSerifSC-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        bold_candidates = [
            bold_font_path,
            os.environ.get("PDF_CJK_BOLD_FONT", ""),
        ]
        body_font = find_first_existing(p for p in font_candidates if p)
        bold_font = find_first_existing(p for p in bold_candidates if p)
        if body_font is None:
            raise SystemExit("未找到可用中文字体。可用 --font /path/to/NotoSansSC-Regular.ttf 指定。")

        self.body = self.fm.FontProperties(fname=str(body_font))
        self.bold = self.fm.FontProperties(fname=str(bold_font or body_font), weight="bold")
        # 代码块里也可能出现中文路径或注释。优先使用同一中文字体，避免 PDF 缺字；
        # 对齐性让位于可打印性和完整显示。
        self.mono = self.fm.FontProperties(fname=str(body_font))

    @property
    def content_w(self) -> float:
        return self.page_w - self.margin_l - self.margin_r

    def close(self) -> None:
        if self.fig is not None:
            self._finish_page()
        self.pdf.close()

    def _new_fig(self) -> None:
        self.fig = self.plt.figure(figsize=(8.27, 11.69), dpi=150)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.page_w)
        self.ax.set_ylim(0, self.page_h)
        self.ax.axis("off")
        self.ax.add_patch(self.Rectangle((0, 0), self.page_w, self.page_h, facecolor=self.paper, edgecolor="none"))

    def _finish_page(self) -> None:
        if self.fig is None:
            return
        self.pdf.savefig(self.fig, bbox_inches="tight", pad_inches=0)
        self.plt.close(self.fig)
        self.fig = None
        self.ax = None

    def new_page(self, header: bool = True) -> None:
        if self.fig is not None:
            self._finish_page()
        self.page_no += 1
        self._new_fig()
        self.y = self.page_h - self.margin_t
        if header:
            self._draw_header_footer()

    def _draw_header_footer(self) -> None:
        assert self.ax is not None
        self.ax.text(self.margin_l, self.page_h - 30, self.title, fontproperties=self.body, fontsize=8.5, color=self.muted, va="top")
        self.ax.plot([self.margin_l, self.page_w - self.margin_r], [self.page_h - 44, self.page_h - 44], color="#d9e2ec", lw=0.6)
        self.ax.text(self.margin_l, 30, self.source_name, fontproperties=self.body, fontsize=7.5, color="#98a2b3", va="bottom")
        self.ax.text(self.page_w - self.margin_r, 30, f"{self.page_no}", fontproperties=self.body, fontsize=8, color="#98a2b3", va="bottom", ha="right")

    def ensure(self, height: float) -> None:
        if self.fig is None:
            self.new_page()
        if self.y - height < self.margin_b:
            self.new_page()

    def text_units_capacity(self, width: float, font_size: float) -> int:
        return max(12, int(width / (font_size * 0.52)))

    def cover(self, subtitle: str) -> None:
        self.new_page(header=False)
        assert self.ax is not None
        self.ax.add_patch(self.Rectangle((0, 0), self.page_w, self.page_h, facecolor="#f8fbff", edgecolor="none"))
        self.ax.add_patch(self.Rectangle((0, 0), 18, self.page_h, facecolor=self.navy, edgecolor="none"))
        self.ax.add_patch(self.Rectangle((18, self.page_h - 190), self.page_w - 18, 190, facecolor="#e8f1fb", edgecolor="none"))
        self.ax.text(62, self.page_h - 118, self.title, fontproperties=self.bold, fontsize=27, color=self.navy, va="top")
        self.ax.text(64, self.page_h - 176, subtitle, fontproperties=self.body, fontsize=13, color=self.blue, va="top")
        self.ax.plot([64, self.page_w - 70], [self.page_h - 224, self.page_h - 224], color=self.blue, lw=1.6)
        notes = [
            "打印版说明",
            "A4 纵向排版，适合纸质审阅和手写标注。",
            "主线入口、候选链路、官方 VM 补丁和实验脚本已分层说明。",
            f"源文件：{self.source_name}",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        y = self.page_h - 292
        for i, line in enumerate(notes):
            size = 16 if i == 0 else 11
            prop = self.bold if i == 0 else self.body
            color = self.navy if i == 0 else self.ink
            self.ax.text(78, y, line, fontproperties=prop, fontsize=size, color=color, va="top")
            y -= 34 if i == 0 else 24
        self.ax.text(self.page_w - 70, 58, "jittor-pointcloud-denoise", fontproperties=self.body, fontsize=9, color="#98a2b3", ha="right")

    def navigation(self, headings: list[Element]) -> None:
        self.new_page()
        self.heading("内容导航", 1)
        shown = [h for h in headings if h.level <= 2]
        for h in shown:
            indent = 0 if h.level == 1 else 18
            bullet = "■" if h.level == 1 else "•"
            self.list_item(f"{bullet} {h.text}", indent=indent, font_size=10.2, marker="")

    def heading(self, text: str, level: int) -> None:
        text = strip_inline_markdown(text)
        if level == 1:
            self.ensure(54)
            assert self.ax is not None
            self.ax.add_patch(self.Rectangle((self.margin_l, self.y - 32), 5, 31, facecolor=self.blue, edgecolor="none"))
            self.ax.text(self.margin_l + 14, self.y, text, fontproperties=self.bold, fontsize=18.5, color=self.navy, va="top")
            self.y -= 46
        elif level == 2:
            self.ensure(38)
            assert self.ax is not None
            self.ax.text(self.margin_l, self.y, text, fontproperties=self.bold, fontsize=14, color=self.blue, va="top")
            self.ax.plot([self.margin_l, self.page_w - self.margin_r], [self.y - 23, self.y - 23], color="#d9e2ec", lw=0.7)
            self.y -= 34
        elif level == 3:
            self.ensure(30)
            assert self.ax is not None
            self.ax.text(self.margin_l, self.y, text, fontproperties=self.bold, fontsize=12.2, color=self.navy, va="top")
            self.y -= 26
        else:
            self.ensure(24)
            assert self.ax is not None
            self.ax.text(self.margin_l, self.y, text, fontproperties=self.bold, fontsize=10.8, color=self.ink, va="top")
            self.y -= 21

    def paragraph(self, text: str) -> None:
        font_size = 10.2
        line_h = 15.2
        max_units = self.text_units_capacity(self.content_w, font_size)
        lines = wrap_display(text, max_units)
        self.ensure(len(lines) * line_h + 8)
        assert self.ax is not None
        for line in lines:
            self.ax.text(self.margin_l, self.y, line, fontproperties=self.body, fontsize=font_size, color=self.ink, va="top")
            self.y -= line_h
        self.y -= 5

    def list_item(self, text: str, indent: float = 0, font_size: float = 10.0, marker: str = "•") -> None:
        x = self.margin_l + indent
        text_x = x + (14 if marker else 0)
        max_units = self.text_units_capacity(self.page_w - self.margin_r - text_x, font_size)
        lines = wrap_display(text, max_units)
        line_h = 14.8
        self.ensure(len(lines) * line_h + 4)
        assert self.ax is not None
        if marker:
            self.ax.text(x, self.y, marker, fontproperties=self.bold, fontsize=font_size, color=self.blue, va="top")
        for idx, line in enumerate(lines):
            self.ax.text(text_x, self.y, line, fontproperties=self.body, fontsize=font_size, color=self.ink, va="top")
            self.y -= line_h
        self.y -= 2

    def bullet_list(self, items: list[str]) -> None:
        for item in items:
            self.list_item(item)
        self.y -= 4

    def code_block(self, text: str) -> None:
        font_size = 8.2
        line_h = 11.2
        pad_x = 9
        pad_y = 7
        max_units = self.text_units_capacity(self.content_w - pad_x * 2, font_size)
        lines: list[str] = []
        for raw in text.splitlines() or [""]:
            if not raw:
                lines.append("")
            else:
                lines.extend(wrap_display(raw.rstrip(), max_units))

        i = 0
        while i < len(lines):
            available = max(1, int((self.y - self.margin_b - pad_y * 2) / line_h))
            if available < 3:
                self.new_page()
                available = max(1, int((self.y - self.margin_b - pad_y * 2) / line_h))
            chunk = lines[i : i + available]
            height = len(chunk) * line_h + pad_y * 2
            self.ensure(height + 6)
            assert self.ax is not None
            top = self.y
            self.ax.add_patch(
                self.Rectangle(
                    (self.margin_l, top - height),
                    self.content_w,
                    height,
                    facecolor=self.code_bg,
                    edgecolor="#d0d7de",
                    lw=0.7,
                )
            )
            y = top - pad_y
            for line in chunk:
                self.ax.text(self.margin_l + pad_x, y, line, fontproperties=self.mono, fontsize=font_size, color="#24292f", va="top")
                y -= line_h
            self.y = top - height - 9
            i += len(chunk)

    def table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        col_count = max(len(r) for r in rows)
        norm_rows = [r + [""] * (col_count - len(r)) for r in rows]
        weights = []
        for c in range(col_count):
            values = [display_units(r[c]) for r in norm_rows[: min(len(norm_rows), 18)]]
            weights.append(max(10, min(34, max(values) + 4)))
        total = sum(weights)
        col_widths = [self.content_w * w / total for w in weights]
        if col_count >= 5:
            font_size = 7.0
        elif col_count == 4:
            font_size = 7.6
        else:
            font_size = 8.4
        line_h = font_size * 1.32
        pad_x = 5
        pad_y = 6

        for row_idx, row in enumerate(norm_rows):
            wrapped_cells: list[list[str]] = []
            for cell, width in zip(row, col_widths):
                max_units = self.text_units_capacity(width - pad_x * 2, font_size)
                wrapped_cells.append(wrap_display(cell, max_units))
            row_h = max(len(c) for c in wrapped_cells) * line_h + pad_y * 2
            self.ensure(row_h + 2)
            assert self.ax is not None
            x = self.margin_l
            y_top = self.y
            is_header = row_idx == 0
            bg = self.navy if is_header else ("#fbfdff" if row_idx % 2 else "#f3f7fb")
            fg = "white" if is_header else self.ink
            for cell_lines, width in zip(wrapped_cells, col_widths):
                self.ax.add_patch(self.Rectangle((x, y_top - row_h), width, row_h, facecolor=bg, edgecolor=self.grid, lw=0.6))
                yy = y_top - pad_y
                for line in cell_lines:
                    self.ax.text(x + pad_x, yy, line, fontproperties=self.bold if is_header else self.body, fontsize=font_size, color=fg, va="top")
                    yy -= line_h
                x += width
            self.y -= row_h
        self.y -= 11

    def render_elements(self, elements: list[Element]) -> None:
        self.new_page()
        for element in elements:
            if element.kind == "heading":
                self.heading(element.text, element.level)
            elif element.kind == "paragraph":
                self.paragraph(element.text)
            elif element.kind == "list" and element.items is not None:
                self.bullet_list(element.items)
            elif element.kind == "code":
                self.code_block(element.text)
            elif element.kind == "table" and element.rows is not None:
                self.table(element.rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PROJECT_ARCHITECTURE_CN.md to a polished A4 PDF.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Markdown input path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="PDF output path")
    parser.add_argument("--font", default="", help="Optional Chinese regular font path")
    parser.add_argument("--bold-font", default="", help="Optional Chinese bold font path")
    parser.add_argument("--title", default="当前项目总架构说明", help="PDF title")
    parser.add_argument("--subtitle", default="Jittor 点云降噪项目架构打印版", help="Cover subtitle")
    parser.add_argument("--no-toc", action="store_true", help="Do not include the navigation page")
    args = parser.parse_args()

    md_path = Path(args.input).resolve()
    pdf_path = Path(args.output).resolve()
    if not md_path.exists():
        raise FileNotFoundError(md_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    elements = parse_markdown(md_path)
    headings = [e for e in elements if e.kind == "heading"]
    renderer = PdfRenderer(
        pdf_path=pdf_path,
        title=args.title,
        source_name=str(md_path.relative_to(ROOT)) if md_path.is_relative_to(ROOT) else str(md_path),
        font_path=Path(args.font).expanduser() if args.font else None,
        bold_font_path=Path(args.bold_font).expanduser() if args.bold_font else None,
    )
    try:
        renderer.cover(args.subtitle)
        if not args.no_toc:
            renderer.navigation(headings)
        renderer.render_elements(elements)
    finally:
        renderer.close()

    print(f"PDF written: {pdf_path}")


if __name__ == "__main__":
    main()
