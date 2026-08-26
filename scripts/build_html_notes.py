"""Convert notes/audio_processing_notes.md into a styled standalone HTML file.

Usage:  python scripts/build_html_notes.py
Output: notes/audio_processing_notes.html
"""

import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "notes" / "audio_processing_notes.md"
HTML_PATH = ROOT / "notes" / "audio_processing_notes.html"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio Signal Processing for Deep Learning — Learner Notes</title>
<script>
  MathJax = {{
    tex: {{
      inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]],
      displayMath: [["$$", "$$"], ["\\\\[", "\\\\]"]]
    }}
  }};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #171a23;
    --surface2: #1e2230;
    --text: #d7dae2;
    --muted: #8b91a3;
    --accent: #7aa2f7;
    --accent2: #9ece6a;
    --border: #2a2f3f;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.7;
    font-size: 16.5px;
  }}
  main {{
    max-width: 880px;
    margin: 0 auto;
    padding: 48px 28px 120px;
  }}
  h1 {{
    font-size: 2.1rem;
    line-height: 1.25;
    color: #fff;
    border-bottom: 3px solid var(--accent);
    padding-bottom: 14px;
  }}
  h2 {{
    font-size: 1.5rem;
    color: var(--accent);
    margin-top: 3.2rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
  }}
  h3 {{ font-size: 1.15rem; color: var(--accent2); margin-top: 2rem; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  blockquote {{
    margin: 1.2em 0;
    padding: 12px 20px;
    background: var(--surface);
    border-left: 4px solid var(--accent2);
    border-radius: 0 8px 8px 0;
    color: var(--text);
  }}
  blockquote p {{ margin: 0.4em 0; }}
  code {{
    font-family: "SF Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.86em;
    background: var(--surface2);
    padding: 2px 6px;
    border-radius: 5px;
    color: #e0af68;
  }}
  pre {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    overflow-x: auto;
    line-height: 1.55;
  }}
  pre code {{ background: none; padding: 0; color: #c0caf5; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1.4em 0;
    font-size: 0.94em;
  }}
  th, td {{
    border: 1px solid var(--border);
    padding: 9px 14px;
    text-align: left;
    vertical-align: top;
  }}
  th {{ background: var(--surface2); color: #fff; }}
  tr:nth-child(even) td {{ background: var(--surface); }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 2.5rem 0; }}
  li {{ margin: 0.3em 0; }}
  strong {{ color: #fff; }}
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1.4em auto;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: #fff;
  }}
  .MathJax {{ color: var(--text); }}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


MATH_PATTERN = re.compile(r"\$\$.+?\$\$|\\\(.+?\\\)", re.DOTALL)


def protect_math(md_text: str) -> tuple[str, list[str]]:
    """Swap LaTeX spans for placeholders so markdown doesn't mangle them."""
    spans: list[str] = []

    def stash(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"MATHPLACEHOLDER{len(spans) - 1}ENDMATH"

    return MATH_PATTERN.sub(stash, md_text), spans


def restore_math(html: str, spans: list[str]) -> str:
    for i, span in enumerate(spans):
        html = html.replace(f"MATHPLACEHOLDER{i}ENDMATH", span)
    return html


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    protected, spans = protect_math(md_text)
    body = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    body = restore_math(body, spans)
    HTML_PATH.write_text(TEMPLATE.format(body=body), encoding="utf-8")
    print(f"Wrote {HTML_PATH} ({HTML_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
