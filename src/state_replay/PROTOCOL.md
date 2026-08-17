# State replay integrator protocol

The integrator and renderer exchange complete snapshots with schema
`tardigrade/state-snapshot/v1`. A snapshot contains `stream_id`, `tick`,
`subtick`, `sequence`, the deterministic `state_hash`, and a complete `entities`
array. Each entity has exactly `id`, `generation`, `class`, and `components`.
Entity IDs remain stable within a lifetime; reusing an ID after destruction
requires a strictly greater generation.

Renderer-facing components use an explicit provenance envelope:

```json
{
  "animation": {
    "provenance": "recorded",
    "value": {"active_node": 17, "cycle": 0.25}
  },
  "ragdoll": {
    "provenance": "unavailable",
    "reason": "not present in the source capture"
  }
}
```

The integrator preserves component JSON without inventing provenance. Capture
normalizers must emit `recorded`, `derived`, or `unavailable`; a renderer must
not guess which one applies. Integrator-only producers may store opaque JSON,
but such snapshots are not renderer-ready until normalized.

Transactions use schema `tardigrade/state-transaction/v1` and contain an exact
next `sequence`, nondecreasing `(tick, subtick)`, the current `base_hash`, and an
ordered nonempty `ops` array. Operations are `create`, `update`, `destroy`, and
`action`. An update deep-merges its component patch. Omission means unchanged,
JSON null is a retained value, and component deletion uses the explicit
`remove_components` array. A supplied `post_hash` is verified.

An action operation records input or another decision at its exact point in
the transaction. It does not itself mutate entity state. `/action-context`
returns the state after preceding operations as `pre` and the complete
transaction state as `post`; consumers must not infer actions from visual
fields such as `camera.value.fire`.

The initial complete checkpoint has sequence zero. An ordinary checkpoint must
equal current state. A history gap, seek, reset, or source change can only be
crossed by installing a complete checkpoint whose sequence and position advance
and whose `discontinuity` is `{kind, reason}`. Deltas cannot cross a gap.

HTTP service endpoints are:

- `POST /manifest`: `{manifest, checkpoint}`
- `POST /ingest`: `{stream_id, transaction}` or an atomic `transactions` array
- `POST /checkpoint`: `{stream_id, checkpoint}`
- `GET /state?stream_id=...&tick=...&subtick=...`
- `POST /batch`: `{stream_id, positions:[{tick,subtick}, ...]}`
- `GET /action-context?stream_id=...&action_id=...`
- `GET /health` and `GET /stats?stream_id=...`

The server requires an explicit artifact directory outside the checkout and
bounds request bytes and batch cardinality. It is a storage/integration service,
not a CS2 process attachment and not a claim of GPU rendering throughput.
