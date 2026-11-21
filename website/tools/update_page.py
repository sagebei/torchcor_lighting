"""Put the generated ECG and pressure-volume data into index.html.

    python update_page.py      # data/ecg.svg and data/em_pv.json -> ../index.html

Replaces the inline <svg class="ecg"> and the <script id="pv-data"> JSON in place; any
input that is missing is skipped. The videos need no page change: they are referenced
by file name from assets/media.
"""
import json
import os
import re
from pathlib import Path

import render as r

PAGE = Path(os.environ.get("TCL_PAGE", r.TOOLS.parent / "index.html"))


def main():
    html = PAGE.read_text()
    svg_file, pv_file = r.DATA / "ecg.svg", r.DATA / "em_pv.json"
    if svg_file.exists():
        i = html.index('<svg class="ecg"')
        j = html.index("</svg>", i) + len("</svg>")
        html = html[:i] + svg_file.read_text() + html[j:]
        print("ECG updated")
    if pv_file.exists():
        data = json.dumps(json.loads(pv_file.read_text()), separators=(",", ":"))
        html, n = re.subn(r'(<script type="application/json" id="pv-data">).*?(</script>)',
                          lambda m: m.group(1) + data + m.group(2), html, flags=re.S)
        print("pressure-volume data updated" if n else "pv-data block not found")
    PAGE.write_text(html)


if __name__ == "__main__":
    main()
