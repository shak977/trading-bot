"""Fast design preview — re-render dashboard.html from the LAST saved snapshot (signals.json),
skipping the slow scan/network entirely. Use this while iterating on the UI:

    python3 preview.py        # writes dashboard.html in ~1s, then: open dashboard.html

It uses whatever data is already in signals.json (the most recent real build), so the layout and
numbers are real — only the styling/markup you're editing changes. For a true data refresh, run the
full `python3 dashboard.py`.
"""
import json
import sys

import dashboard

try:
    snap = json.load(open("signals.json"))
except Exception as e:  # noqa: BLE001
    print(f"[preview] couldn't read signals.json ({e}). Run `python3 dashboard.py` once first.")
    sys.exit(1)

with open("dashboard.html", "w") as f:
    f.write(dashboard.render_html(snap))
print(f"[preview] dashboard.html re-rendered from signals.json @ {snap.get('generated_at','?')}")
