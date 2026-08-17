"""Boring REST client: health-check then POST /render, print the manifest."""
import argparse
import json
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="render service base URL")
    ap.add_argument(
        "request",
        help="JSON request containing profile plus logstream (cs2) or snapshot (csgo_legacy)",
    )
    args = ap.parse_args()
    try:
        health = json.loads(urllib.request.urlopen(
            args.base + "/health", timeout=5).read())
        variants = {
            value["id"] for value in health.get("game_variants", [])
            if isinstance(value, dict) and isinstance(value.get("id"), str)
        }
        if not variants:
            raise SystemExit(
                "render service does not advertise explicit game-variant support"
            )
        print("GET /health -> variants", ",".join(sorted(variants)))
        with open(args.request, encoding="utf-8") as fh:
            request_document = json.load(fh)
        requested_variant = request_document.get("game_variant")
        if requested_variant is not None and requested_variant not in variants:
            raise SystemExit(
                f"request game_variant {requested_variant!r} is not advertised "
                f"by the service: {sorted(variants)!r}"
            )
        body = json.dumps(request_document).encode()
        request = urllib.request.Request(
            args.base + "/render", data=body,
            headers={"Content-Type": "application/json"})
        response = json.loads(
            urllib.request.urlopen(request, timeout=3600).read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"render service HTTP {exc.code}: {detail}") from None

    selected_variant = response.get("game_variant")
    if selected_variant not in variants:
        raise SystemExit(
            f"render response omitted or returned an unadvertised game_variant: "
            f"{selected_variant!r}"
        )
    if requested_variant is not None and selected_variant != requested_variant:
        raise SystemExit(
            "render response game_variant disagrees with the request: "
            f"{selected_variant!r} != {requested_variant!r}"
        )
    if selected_variant == "cs2":
        print("POST /render -> variant", selected_variant, "job",
              response.get("job_id"), "match_id",
              response["match_id"], "players", response["n_players"], "res",
              response["resolution"], "stride", response["stride"])
        for view, result in response["views"].items():
            steered = sum(p.get("n_future_steered", 0)
                          for p in result.get("players", []))
            print("  view %-5s rc=%s outputs=%d future_steered_total=%d"
                  % (view, result.get("rc"), len(result.get("outputs", [])),
                     steered))
    else:
        print("POST /render -> variant", selected_variant, "job",
              response.get("job_id"), "snapshot", response.get("snapshot"),
              "res", response.get("resolution"), "outputs",
              ",".join(response.get("outputs", [])))


if __name__ == "__main__":
    main()
