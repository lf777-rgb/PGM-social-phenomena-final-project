#!/usr/bin/env python3
"""Render a static HTML version of the manuscript without requiring Quarto."""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

import mistune


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS_IMAGES = DOCS / "images"


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    _, raw_yaml, body = text.split("---", 2)
    meta: dict[str, str] = {}
    current_key = None
    for line in raw_yaml.splitlines():
        if not line.strip():
            continue
        if re.match(r"^[A-Za-z_-]+:", line):
            key, value = line.split(":", 1)
            current_key = key.strip()
            meta[current_key] = value.strip().strip('"')
        elif current_key:
            meta[current_key] += "\n" + line
    return meta, body.strip()


def extract_abstract(meta: dict[str, str]) -> str:
    raw = meta.get("abstract", "")
    raw = raw.replace("|", "").strip()
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines)


def preprocess_markdown(body: str) -> str:
    def latex_to_text(source: str) -> str:
        text = " ".join(source.split())
        text = text.replace(
            r"\frac{n(x,a)+\alpha}{n(a)+\alpha |\mathcal{X}|}",
            "(n(x,a)+alpha) / (n(a)+alpha*|X|)",
        )
        text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", text)
        replacements = {
            r"\times": "*",
            r"\sim": "~",
            r"\mid": " given ",
            r"\in": "in",
            r"\hat{P}": "P_hat",
            r"\mathcal{N}": "Normal",
            r"\mathcal{X}": "X",
            r"\pi": "pi",
            r"\mu": "mu",
            r"\Sigma": "Sigma",
            r"\alpha": "alpha",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace("{", "").replace("}", "")
        text = text.replace("\\", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def display_math(match: re.Match[str]) -> str:
        equation = latex_to_text(match.group(1))
        return f'\n<div class="equation">{html.escape(equation)}</div>\n'

    def inline_math(match: re.Match[str]) -> str:
        return f'<span class="math-inline">{html.escape(latex_to_text(match.group(1)))}</span>'

    # Convert Quarto image attributes into plain HTML figures for this static render.
    def repl(match: re.Match[str]) -> str:
        alt = match.group(1)
        src = match.group(2)
        return f'\n<figure><img src="{src}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>\n'

    body = re.sub(r"\$\$(.*?)\$\$", display_math, body, flags=re.S)
    body = re.sub(r"\$([^$\n]+)\$", inline_math, body)
    body = re.sub(r"!\[([^\]]+)\]\(([^)]+)\)\{[^}]*\}", repl, body)
    body = re.sub(r"\{\.unnumbered\}", "", body)
    return body


def copy_images() -> None:
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    for name in [
        "pgm_dag.svg",
        "profile_summary.svg",
        "ev_vs_ports.svg",
        "charging_cpt_heatmap.svg",
    ]:
        shutil.copy2(ROOT / "images" / name, DOCS_IMAGES / name)


def main() -> None:
    meta, body = split_front_matter((ROOT / "index.qmd").read_text())
    copy_images()
    markdown = mistune.create_markdown(escape=False, plugins=["table"])
    content = markdown(preprocess_markdown(body))
    title = meta.get("title", "Final Project")
    subtitle = meta.get("subtitle", "")
    abstract_text = extract_abstract(meta)
    rendered = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root {{
      --ink: #17212f;
      --muted: #5e6a78;
      --rule: #d7dde5;
      --banner: #041e42;
      --paper: #ffffff;
      --soft: #f6f8fb;
      --accent: #2f6f8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font: 17px/1.62 Georgia, "Times New Roman", serif;
    }}
    header {{
      background: var(--banner);
      color: #f3f5f8;
      padding: 48px max(24px, calc((100vw - 920px) / 2)) 38px;
    }}
    header h1 {{
      margin: 0 0 12px;
      max-width: 920px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: clamp(2rem, 4vw, 3.8rem);
      line-height: 1.08;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      color: #d9e2ec;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .abstract {{
      max-width: 920px;
      margin: 24px auto 0;
      padding: 18px 20px;
      background: rgba(255,255,255,0.08);
      border-left: 4px solid #86c5da;
      font-size: 0.98rem;
    }}
    main {{
      max-width: 920px;
      margin: 42px auto 80px;
      padding: 0 24px;
    }}
    h2, h3 {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.25;
      margin-top: 2.2em;
      letter-spacing: 0;
    }}
    h2 {{
      border-bottom: 1px solid var(--rule);
      padding-bottom: 8px;
    }}
    a {{ color: var(--accent); }}
    blockquote {{
      border-left: 4px solid var(--accent);
      margin: 22px 0;
      padding: 6px 0 6px 18px;
      color: #263548;
      background: var(--soft);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0 30px;
      font-size: 0.95rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--rule);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--soft);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    figure {{
      margin: 30px 0;
    }}
    figure img {{
      display: block;
      max-width: 100%;
      height: auto;
      border: 1px solid var(--rule);
      background: #fff;
    }}
    .equation {{
      margin: 18px auto;
      padding: 12px 16px;
      max-width: 760px;
      overflow-x: auto;
      text-align: center;
      background: #f7f9fc;
      border: 1px solid var(--rule);
      border-radius: 6px;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.95rem;
    }}
    .math-inline {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.92em;
      background: #f7f9fc;
      padding: 0 2px;
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.92rem;
      font-style: italic;
    }}
    code, pre {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.92em;
    }}
    pre {{
      overflow-x: auto;
      padding: 14px;
      background: #f1f4f8;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(subtitle)}</p>
    <p>Lanbin Fan - Georgetown University, Data Science and Analytics - August 10, 2026</p>
    <div class="abstract"><strong>Abstract.</strong> {html.escape(abstract_text)}</div>
  </header>
  <main>
    {content}
  </main>
</body>
</html>
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(rendered)
    print(DOCS / "index.html")


if __name__ == "__main__":
    main()
