"""Build index.html: inline tensors.json into template.html. Run after export_tensors.py."""
import json, os
d = os.path.dirname(os.path.abspath(__file__))
data = json.dumps(json.load(open(os.path.join(d, "tensors.json"))), separators=(",", ":"), ensure_ascii=False)
data = data.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
t = open(os.path.join(d, "template.html")).read()
assert "/*__DATA__*/null" in t
open(os.path.join(d, "index.html"), "w").write(t.replace("/*__DATA__*/null", data))
print("wrote index.html", os.path.getsize(os.path.join(d, "index.html")), "bytes")
