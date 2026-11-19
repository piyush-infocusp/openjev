"""Build samples/index.html from samples/*.json (one page, data embedded)."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
GROUPS = [("games", "Games"), ("strategy", "Strategy & forecasting"),
          ("robotics", "Robotics & embodied"), ("workflow", "Workflow, tools & rating")]

data = {}
for key, _ in GROUPS:
    p = HERE / f"{key}.json"
    data[key] = json.loads(p.read_text()) if p.exists() else []

page = (HERE / "page_template.html").read_text()
page = page.replace("/*GROUPS*/", json.dumps(GROUPS)).replace("/*DATA*/", json.dumps(data).replace("</", "<\\/"))
(HERE / "index.html").write_text(page)
print("wrote", HERE / "index.html", {k: len(v) for k, v in data.items()})
