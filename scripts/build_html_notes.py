"""Build the interactive HTML version of the notes.

Usage:  python scripts/build_html_notes.py
Input:  notes/<MD_NAME>.md   +  scripts/notes_widgets.js
Output: notes/<MD_NAME>.html  (single self-contained page; figures referenced relatively)

Markdown conventions understood beyond standard markdown:
  <!-- widget: NAME -->            mounts the interactive widget NAME (see notes_widgets.js)
  > **Key takeaway** ...           styled callout (also: Researcher's corner, Industry corner,
                                   Try it, Common mistake, Deep dive)
  <details><summary>..</summary>   collapsible quiz / answer reveal (works on GitHub too)
  $$...$$ and \\(...\\)             MathJax
"""

import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent

# ---- module configuration -------------------------------------------------
TITLE = "Audio Signal Processing for Deep Learning"
SUBTITLE = "Interactive learner notes — waveforms, Fourier, spectrograms, mel, Griffin-Lim"
MD_PATH = ROOT / "notes" / "audio_processing_notes.md"
HTML_PATH = ROOT / "notes" / "audio_processing_notes.html"
WIDGETS_JS = ROOT / "scripts" / "notes_widgets.js"
# ---------------------------------------------------------------------------

CALLOUTS = {
    "Key takeaway": "takeaway",
    "Researcher's corner": "research",
    "Industry corner": "industry",
    "Try it": "tryit",
    "Common mistake": "mistake",
    "Deep dive": "deep",
}

MATH_PATTERN = re.compile(r"\$\$.+?\$\$|\\\(.+?\\\)", re.DOTALL)
WIDGET_PATTERN = re.compile(r"<!--\s*widget:\s*([a-z0-9\-]+)\s*-->")


def protect_math(text: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"MATHPLACEHOLDER{len(spans) - 1}ENDMATH"

    return MATH_PATTERN.sub(stash, text), spans


def restore_math(body: str, spans: list[str]) -> str:
    for i, s in enumerate(spans):
        body = body.replace(f"MATHPLACEHOLDER{i}ENDMATH", s)
    return body


def mount_widgets(text: str) -> str:
    return WIDGET_PATTERN.sub(
        lambda m: f'\n\n<div class="widget" data-widget="{m.group(1)}"></div>\n\n', text)


def style_callouts(body: str) -> str:
    """`> **Key takeaway** — text` blockquotes -> styled callout boxes."""
    # python-markdown merges consecutive blockquotes into one; split them back
    # apart wherever a new labelled paragraph starts.
    labels = "|".join(re.escape(l) for l in CALLOUTS)
    body = re.sub(rf"</p>\s*<p><strong>({labels})", r"</p></blockquote>\n<blockquote><p><strong>\1", body)
    for label, cls in CALLOUTS.items():
        pattern = (rf"<blockquote>\s*<p><strong>{re.escape(label)}[^<]*</strong>"
                   rf"\s*(?:—|–|-|:)?\s*")
        body = re.sub(
            pattern,
            f'<blockquote class="callout {cls}"><p><span class="callout-label">{html.escape(label)}</span>',
            body,
        )
    return body


def build_toc(body: str) -> str:
    items = []
    for m in re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', body, re.DOTALL):
        text = re.sub(r"<[^>]+>", "", m.group(2))
        items.append(f'<li><a href="#{m.group(1)}" data-target="{m.group(1)}">'
                     f'<input type="checkbox" class="done" data-section="{m.group(1)}" title="mark as done">'
                     f'<span>{text}</span></a></li>')
    return "\n".join(items)


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    md_text = mount_widgets(md_text)
    protected, spans = protect_math(md_text)
    body = markdown.markdown(
        protected,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "md_in_html", "attr_list"],
    )
    body = restore_math(body, spans)
    body = style_callouts(body)
    toc = build_toc(body)
    widgets_js = WIDGETS_JS.read_text(encoding="utf-8")

    page = (TEMPLATE
            .replace("{{TITLE}}", html.escape(TITLE))
            .replace("{{SUBTITLE}}", html.escape(SUBTITLE))
            .replace("{{TOC}}", toc)
            .replace("{{BODY}}", body)
            .replace("{{WIDGETS_JS}}", widgets_js)
            .replace("{{STORAGE_KEY}}", MD_PATH.stem))
    HTML_PATH.write_text(page, encoding="utf-8")
    n_widgets = len(WIDGET_PATTERN.findall(MD_PATH.read_text(encoding="utf-8")))
    print(f"Wrote {HTML_PATH} ({HTML_PATH.stat().st_size / 1024:.0f} KB, "
          f"{toc.count('<li>')} sections, {n_widgets} widgets)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}} — Interactive Notes</title>
<script>
  MathJax = { tex: { inlineMath: [["\\(", "\\)"], ["$", "$"]], displayMath: [["$$", "$$"], ["\\[", "\\]"]] } };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/styles/github-dark.min.css">
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/highlight.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --surface: #171a23; --surface2: #1e2230; --text: #d7dae2; --muted: #8b91a3;
    --accent: #7aa2f7; --accent2: #9ece6a; --orange: #ff9e64; --red: #f7768e; --purple: #bb9af7; --yellow: #e0af68;
    --border: #2a2f3f; --sidebar-w: 300px;
  }
  [data-theme="light"] {
    --bg: #f7f8fb; --surface: #ffffff; --surface2: #eef1f7; --text: #1d2230; --muted: #5c6478;
    --accent: #2f6feb; --accent2: #3a8f3a; --orange: #d9730d; --red: #d0405f; --purple: #7c4dcc; --yellow: #b07d10;
    --border: #d9deea;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; background: var(--bg); color: var(--text); line-height: 1.7; font-size: 16.5px;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
  #progress { position: fixed; top: 0; left: 0; height: 3px; width: 0; background: linear-gradient(90deg, var(--accent), var(--accent2)); z-index: 50; transition: width .1s; }

  /* layout */
  .layout { display: grid; grid-template-columns: var(--sidebar-w) 1fr; min-height: 100vh; }
  aside { position: sticky; top: 0; height: 100vh; overflow-y: auto; background: var(--surface); border-right: 1px solid var(--border); padding: 22px 18px; }
  aside h1 { font-size: 1rem; margin: 0 0 4px; color: var(--text); line-height: 1.3; }
  aside .sub { font-size: .8rem; color: var(--muted); margin: 0 0 14px; }
  aside .stats { font-size: .78rem; color: var(--muted); margin: 8px 0 14px; padding: 8px 10px; background: var(--surface2); border-radius: 8px; }
  aside .stats b { color: var(--accent2); }
  aside ul { list-style: none; padding: 0; margin: 0; }
  aside li a { display: flex; gap: 8px; align-items: flex-start; padding: 6px 8px; border-radius: 7px; color: var(--muted); text-decoration: none; font-size: .86rem; line-height: 1.35; }
  aside li a:hover { background: var(--surface2); color: var(--text); }
  aside li a.active { background: var(--surface2); color: var(--accent); font-weight: 600; }
  aside input.done { margin: 4px 0 0; accent-color: var(--accent2); flex: none; }
  aside li a.completed span { text-decoration: line-through; opacity: .65; }
  .toolbar { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
  .tbtn { border: 1px solid var(--border); background: var(--surface2); color: var(--text); border-radius: 7px; padding: 5px 10px; font-size: .78rem; cursor: pointer; }
  .tbtn:hover { border-color: var(--accent); }
  #menu-btn { display: none; position: fixed; top: 12px; left: 12px; z-index: 60; }

  main { max-width: 900px; margin: 0 auto; padding: 40px 34px 140px; min-width: 0; }
  @media (max-width: 1000px) {
    .layout { grid-template-columns: 1fr; }
    aside { position: fixed; left: 0; top: 0; width: min(88vw, 340px); transform: translateX(-100%); transition: transform .2s; z-index: 55; }
    aside.open { transform: none; box-shadow: 0 0 40px rgba(0,0,0,.5); }
    #menu-btn { display: block; }
    main { padding: 60px 18px 120px; }
  }

  /* typography */
  h1 { font-size: 2.1rem; line-height: 1.25; border-bottom: 3px solid var(--accent); padding-bottom: 14px; }
  h2 { font-size: 1.5rem; color: var(--accent); margin-top: 3.2rem; padding-top: 1.2rem; border-top: 1px solid var(--border); scroll-margin-top: 20px; }
  h3 { font-size: 1.15rem; color: var(--accent2); margin-top: 2rem; }
  h4 { font-size: 1rem; color: var(--text); margin-top: 1.6rem; }
  a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
  blockquote { margin: 1.2em 0; padding: 12px 20px; background: var(--surface); border-left: 4px solid var(--accent2); border-radius: 0 8px 8px 0; }
  blockquote p { margin: .4em 0; }
  code { font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; font-size: .86em; background: var(--surface2); padding: 2px 6px; border-radius: 5px; color: var(--yellow); }
  pre { position: relative; background: #0b0e14; border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; overflow-x: auto; line-height: 1.55; }
  pre code { background: none; padding: 0; color: #c0caf5; }
  pre .copy { position: absolute; top: 8px; right: 8px; font-size: .72rem; border: 1px solid var(--border); background: var(--surface2); color: var(--text); border-radius: 6px; padding: 3px 8px; cursor: pointer; opacity: 0; transition: opacity .15s; }
  pre:hover .copy { opacity: 1; }
  table { border-collapse: collapse; width: 100%; margin: 1.4em 0; font-size: .94em; display: block; overflow-x: auto; }
  th, td { border: 1px solid var(--border); padding: 9px 14px; text-align: left; vertical-align: top; }
  th { background: var(--surface2); } tr:nth-child(even) td { background: var(--surface); }
  hr { border: none; border-top: 1px solid var(--border); margin: 2.5rem 0; }
  li { margin: .3em 0; } strong { color: var(--text); font-weight: 700; }
  img { max-width: 100%; height: auto; display: block; margin: 1.4em auto; border-radius: 10px; border: 1px solid var(--border); background: #fff; cursor: zoom-in; }
  .MathJax { color: var(--text); }

  /* callouts */
  .callout { border-left-width: 4px; padding: 14px 20px 12px; position: relative; }
  .callout-label { display: inline-block; font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 700; margin-right: 10px; padding: 2px 8px; border-radius: 999px; vertical-align: middle; }
  .callout.takeaway { border-color: var(--accent2); background: color-mix(in srgb, var(--accent2) 8%, var(--surface)); } .callout.takeaway .callout-label { background: var(--accent2); color: #0b0e14; }
  .callout.research { border-color: var(--purple); background: color-mix(in srgb, var(--purple) 8%, var(--surface)); } .callout.research .callout-label { background: var(--purple); color: #0b0e14; }
  .callout.industry { border-color: var(--orange); background: color-mix(in srgb, var(--orange) 8%, var(--surface)); } .callout.industry .callout-label { background: var(--orange); color: #0b0e14; }
  .callout.tryit { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 8%, var(--surface)); } .callout.tryit .callout-label { background: var(--accent); color: #0b0e14; }
  .callout.mistake { border-color: var(--red); background: color-mix(in srgb, var(--red) 8%, var(--surface)); } .callout.mistake .callout-label { background: var(--red); color: #0b0e14; }
  .callout.deep { border-color: var(--yellow); background: color-mix(in srgb, var(--yellow) 8%, var(--surface)); } .callout.deep .callout-label { background: var(--yellow); color: #0b0e14; }

  /* quizzes */
  details { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 16px; margin: .8em 0; }
  details summary { cursor: pointer; font-weight: 600; color: var(--text); }
  details summary::marker { color: var(--accent); }
  details[open] { border-color: var(--accent); }
  details[open] summary { margin-bottom: 8px; color: var(--accent); }

  /* widgets */
  .widget { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px 14px; margin: 1.6em 0; box-shadow: 0 6px 30px rgba(0,0,0,.15); }
  .w-title { font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 10px; }
  .w-badge { font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; background: var(--accent); color: #0b0e14; padding: 2px 8px; border-radius: 999px; }
  .w-help { color: var(--muted); font-size: .88rem; margin: 4px 0 12px; line-height: 1.5; }
  .w-controls { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-bottom: 10px; align-items: center; }
  .w-group { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; padding: 6px 10px; border: 1px dashed var(--border); border-radius: 8px; }
  .w-group-title { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
  .w-slider { display: flex; align-items: center; gap: 8px; font-size: .86rem; }
  .w-label { color: var(--muted); min-width: 0; white-space: nowrap; }
  .w-slider input[type=range] { width: 150px; accent-color: var(--accent); }
  .w-val { font-family: ui-monospace, Menlo, monospace; font-size: .82rem; color: var(--accent2); min-width: 56px; }
  .w-num, .w-select { background: var(--surface2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: .86rem; width: 110px; }
  .w-toggle { display: flex; align-items: center; gap: 6px; font-size: .86rem; } .w-toggle input { accent-color: var(--accent2); }
  .w-btn { border: 1px solid var(--border); background: var(--surface2); color: var(--text); border-radius: 7px; padding: 5px 12px; font-size: .84rem; cursor: pointer; }
  .w-btn:hover { border-color: var(--accent); } .w-btn.primary { background: var(--accent); color: #0b0e14; border-color: var(--accent); }
  .w-canvas { width: 100%; display: block; border-radius: 8px; }
  .w-readout { font-size: .88rem; margin-top: 10px; line-height: 1.55; color: var(--text); }
  .w-readout .muted, .muted { color: var(--muted); }
  .w-readout .good, .good { color: var(--accent2); font-weight: 700; } .w-readout .bad, .bad { color: var(--red); font-weight: 700; }
  .w-table { display: table; width: auto; font-size: .86rem; margin: 6px 0; } .w-table td { padding: 4px 12px; border: none; border-bottom: 1px solid var(--border); }
  .w-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .w-chip { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 4px 8px; font-family: ui-monospace, Menlo, monospace; font-size: .8rem; text-align: center; min-width: 40px; }
  .w-chip small { display: block; color: var(--muted); font-size: .68rem; } .w-chip.special { border-color: var(--orange); color: var(--orange); }
  .w-textarea, .w-text { width: 100%; background: var(--surface2); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: .92rem; font-family: inherit; }

  /* lightbox + back to top */
  #lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.88); display: none; align-items: center; justify-content: center; z-index: 100; cursor: zoom-out; }
  #lightbox img { max-width: 96vw; max-height: 94vh; border: none; background: #fff; }
  #totop { position: fixed; right: 22px; bottom: 22px; z-index: 40; display: none; }
</style>
</head>
<body>
<div id="progress"></div>
<button class="tbtn" id="menu-btn">☰ contents</button>
<div class="layout">
<aside id="sidebar">
  <h1>{{TITLE}}</h1>
  <p class="sub">{{SUBTITLE}}</p>
  <div class="toolbar">
    <button class="tbtn" id="theme-btn">☾ theme</button>
    <button class="tbtn" id="reset-btn" title="clear progress">reset progress</button>
  </div>
  <div class="stats">Progress: <b id="pct">0%</b> · <span id="done-count">0</span> sections done · tick a box when a section makes sense to you</div>
  <ul id="toc">{{TOC}}</ul>
</aside>
<main>
{{BODY}}
</main>
</div>
<div id="lightbox"><img alt=""></div>
<button class="tbtn" id="totop">↑ top</button>

<script>
{{WIDGETS_JS}}
</script>
<script>
(function () {
  const KEY = "notes-progress-{{STORAGE_KEY}}";
  const state = JSON.parse(localStorage.getItem(KEY) || "{}");
  const boxes = Array.from(document.querySelectorAll("aside input.done"));
  const links = Array.from(document.querySelectorAll("aside li a"));
  function save() { localStorage.setItem(KEY, JSON.stringify(state)); }
  function refresh() {
    let n = 0;
    boxes.forEach(b => { const on = !!state[b.dataset.section]; b.checked = on; b.closest("a").classList.toggle("completed", on); if (on) n++; });
    document.getElementById("pct").textContent = Math.round(100 * n / Math.max(1, boxes.length)) + "%";
    document.getElementById("done-count").textContent = n;
  }
  boxes.forEach(b => b.addEventListener("click", e => { e.stopPropagation(); state[b.dataset.section] = b.checked; save(); refresh(); }));
  document.getElementById("reset-btn").addEventListener("click", () => { for (const k in state) delete state[k]; save(); refresh(); });
  refresh();

  // theme
  const themeKey = "notes-theme";
  const applyTheme = t => { document.documentElement.dataset.theme = t; document.getElementById("theme-btn").textContent = t === "dark" ? "☀ light" : "☾ dark"; };
  applyTheme(localStorage.getItem(themeKey) || "dark");
  document.getElementById("theme-btn").addEventListener("click", () => { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; localStorage.setItem(themeKey, t); applyTheme(t); });

  // mobile menu
  const sidebar = document.getElementById("sidebar");
  document.getElementById("menu-btn").addEventListener("click", () => sidebar.classList.toggle("open"));
  sidebar.querySelectorAll("a").forEach(a => a.addEventListener("click", () => sidebar.classList.remove("open")));

  // scroll spy + progress + back-to-top
  const headings = Array.from(document.querySelectorAll("main h2[id]"));
  const totop = document.getElementById("totop");
  function onScroll() {
    const y = window.scrollY, h = document.documentElement.scrollHeight - window.innerHeight;
    document.getElementById("progress").style.width = (100 * y / Math.max(1, h)) + "%";
    totop.style.display = y > 600 ? "block" : "none";
    let current = headings[0];
    for (const hd of headings) { if (hd.getBoundingClientRect().top <= 120) current = hd; }
    links.forEach(a => a.classList.toggle("active", current && a.dataset.target === current.id));
    const act = document.querySelector("aside a.active");
    if (act) { const r = act.getBoundingClientRect(), s = sidebar.getBoundingClientRect(); if (r.top < s.top + 60 || r.bottom > s.bottom - 60) act.scrollIntoView({ block: "center" }); }
  }
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  totop.addEventListener("click", () => window.scrollTo({ top: 0 }));

  // code: highlight + copy buttons
  if (window.hljs) document.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
  document.querySelectorAll("pre").forEach(pre => {
    const b = document.createElement("button"); b.className = "copy"; b.textContent = "copy";
    b.addEventListener("click", () => { navigator.clipboard.writeText(pre.querySelector("code").innerText).then(() => { b.textContent = "copied"; setTimeout(() => b.textContent = "copy", 1200); }); });
    pre.appendChild(b);
  });

  // image lightbox
  const lb = document.getElementById("lightbox"), lbImg = lb.querySelector("img");
  document.querySelectorAll("main img").forEach(img => img.addEventListener("click", () => { lbImg.src = img.src; lb.style.display = "flex"; }));
  lb.addEventListener("click", () => lb.style.display = "none");
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
