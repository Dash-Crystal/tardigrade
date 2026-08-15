"""Boring REST client: health-check then POST /render, print the manifest."""
import json, sys, urllib.request
base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8815"
print("GET /health ->", urllib.request.urlopen(base + "/health", timeout=5).read().decode())
body = json.dumps(json.load(open(sys.argv[2]))).encode()
req = urllib.request.Request(base + "/render", data=body,
                             headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
print("POST /render -> match_id", r["match_id"], "players", r["n_players"],
      "res", r["resolution"], "stride", r["stride"])
for v, d in r["views"].items():
    steered = sum(p.get("n_future_steered", 0) for p in d.get("players", []))
    print("  view %-5s rc=%s tensors=%d future_steered_total=%d"
          % (v, d.get("rc"), len(d.get("tensors", [])), steered))
