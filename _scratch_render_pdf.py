import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

REPORT_DIR = Path("report").resolve()
FIGURES_DIR = Path("figures").resolve()

md_text = Path("report/report.md").read_text(encoding="utf-8")


def ensure_blank_before_lists(text):
    """python-markdown requires a blank line before a list block, or it lazily
    merges '- item' lines into the preceding paragraph as literal hyphens."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        is_list_item = line.startswith("- ") or line.startswith("* ")
        prev_is_list_or_blank = out and (out[-1].strip() == "" or out[-1].startswith("- ") or out[-1].startswith("* "))
        if is_list_item and not prev_is_list_or_blank:
            out.append("")
        out.append(line)
    return "\n".join(out)


md_text = ensure_blank_before_lists(md_text)


def fix_img(m):
    rel = m.group(1)
    abs_path = (REPORT_DIR / rel).resolve()
    return f'src="{abs_path.as_uri()}"'


md_text = re.sub(r'src="(\.\./figures/[^"]+)"', fix_img, md_text)

body_html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm;
}
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1 {
    font-size: 18pt;
    color: #10336b;
    margin-top: 22px;
    border-bottom: 1.5pt solid #10336b;
    padding-bottom: 4px;
}
h2 {
    font-size: 13.5pt;
    color: #1a4d8c;
    margin-top: 16px;
}
h3 { font-size: 11.5pt; color: #333333; }
p { text-align: justify; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0 12px 0;
    font-size: 8pt;
    table-layout: fixed;
}
th, td {
    border: 0.75pt solid #999999;
    padding: 3px 5px;
    text-align: left;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
th { background-color: #e8eef7; }
.figure { text-align: center; margin: 10px 0; }
.figure img { max-width: 100%; }
.titlemeta { font-size: 9.5pt; color: #444444; margin-top: 10px; }
code { font-family: Courier, monospace; font-size: 9pt; background-color: #f2f2f2; padding: 1px 3px; }
strong { color: #10336b; }
"""

html_doc = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{body_html}</body>
</html>"""

Path("report/report.html").write_text(html_doc, encoding="utf-8")

with open("report/report.pdf", "wb") as f:
    result = pisa.CreatePDF(html_doc, dest=f, encoding="utf-8")

print("PDF generation error state:", result.err)
print("wrote report/report.pdf" if not result.err else "PDF GENERATION FAILED")
