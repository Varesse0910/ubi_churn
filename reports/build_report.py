"""Build the churn analysis report: churn_analysis.qmd -> data/08_reporting/churn_analysis.html.

Steps: `quarto convert` (qmd -> ipynb), execute the notebook with nbclient (from reports/, so the
code finds src/ and the catalog), then `quarto render` the executed notebook (Quarto does not
re-execute an .ipynb). Executing through nbclient avoids needing Quarto to locate Python itself.

    python reports/build_report.py                      # quarto on PATH
    python reports/build_report.py --quarto C:/path/to/quarto.exe
    QUARTO=C:/path/to/quarto.exe python reports/build_report.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbclient
import nbformat

REPORT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = REPORT_DIR.parent
SOURCE: Path = REPORT_DIR / 'churn_analysis.qmd'
OUTPUT: Path = PROJECT_ROOT / 'data' / '08_reporting' / 'churn_analysis.html'


def find_quarto(explicit: str | None) -> str:
    """Quarto executable: --quarto, then $QUARTO, then PATH."""
    quarto = explicit or os.environ.get('QUARTO') or shutil.which('quarto')
    if not quarto:
        sys.exit('quarto not found: install it (https://quarto.org) or pass --quarto / set QUARTO')
    return quarto


def build(quarto: str) -> Path:
    """Convert, execute and render the report; return the HTML path."""
    notebook = SOURCE.with_suffix('.ipynb')
    subprocess.run([quarto, 'convert', SOURCE.name, '--output', notebook.name], cwd=REPORT_DIR, check=True)
    nb = nbformat.read(notebook, as_version=4)
    start = time.time()
    nbclient.NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(REPORT_DIR)}}).execute()
    nbformat.write(nb, notebook)
    print(f'executed in {time.time() - start:.0f} s')
    try:
        subprocess.run([quarto, 'render', notebook.name, '--to', 'html'], cwd=REPORT_DIR, check=True)
    finally:
        notebook.unlink(missing_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(REPORT_DIR / 'churn_analysis.html'), OUTPUT)
    shutil.rmtree(REPORT_DIR / 'churn_analysis_files', ignore_errors=True)
    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quarto', help='path to the quarto executable')
    path = build(find_quarto(parser.parse_args().quarto))
    print(f'report -> {path} ({path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
