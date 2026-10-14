"""Render report.md to PDF via HTML and a headless Chromium browser (Edge or Chrome).

Usage (from assignment-1/):
    .venv\\Scripts\\python report\\build_pdf.py
"""

import os
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "report.md")
HTML = os.path.join(HERE, "report.html")
PDF = os.path.join(HERE, "Assignment01_LopezGuillen_Report.pdf")

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; margin: 0 0 8pt; line-height: 1.25; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p, li { text-align: justify; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #e8eef5; }
img { display: block; max-width: 85%; margin: 8pt auto 2pt; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f2f2; padding: 0 2pt; }
hr { border: none; border-top: 1px solid #bbb; }
"""


def main():
    with open(SOURCE, encoding="utf-8") as handle:
        body = markdown.markdown(handle.read(), extensions=["tables"])
    with open(HTML, "w", encoding="utf-8") as handle:
        handle.write(
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>Assignment 1 Report</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>"
        )
    browser = next((path for path in BROWSERS if os.path.exists(path)), None)
    if browser is None:
        sys.exit("No Edge or Chrome installation found; open report.html and print it to PDF manually.")
    subprocess.run(
        [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF}", "file:///" + HTML.replace("\\", "/")],
        check=True,
        capture_output=True,
    )
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
