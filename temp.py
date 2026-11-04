from pathlib import Path
import json

for f in Path("./DEV/aiclub_ics_uci_edu").rglob('*.json'):
    json_obj = json.loads(f.read_text(encoding="utf-8"))
    print(sorted(list(json_obj.keys())))
    # looks like all JSON objects have url, content, encoding