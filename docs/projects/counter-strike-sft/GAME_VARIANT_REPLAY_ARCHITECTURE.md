# Counter-Strike game-variant replay architecture

## Stable boundary

`tardigrade/state-snapshot/v1` and `tardigrade/state-transaction/v1` are the
engine-neutral boundary. The state journal, checkpointing, lifecycle generation,
hash chaining, seeking, batching, and action-context implementation are shared by
all game variants.

The supported variant identifiers are:

- `cs2`: Source 2 captures and assets;
- `csgo_legacy`: the frozen Source 1 CS:GO capture and asset formats.

Aliases, filename inference, and asset fallback across variants are forbidden at
the service boundary. A render profile names exactly one variant. A request may
omit the variant only when it inherits the explicit profile value; an explicit
request/profile mismatch is a refusal.

## Variant-owned work

The capture normalizer owns demo framing, network-message decoding, entity
baselines, field paths, and the decision to label each component `recorded`,
`derived`, or `unavailable`. A Source 1 normalizer must never describe a
client-computed animation pose as recorded merely because it decoded sequence,
cycle, pose parameters, or animation layers.

The renderer backend owns map, model, material, skeleton, animation, particle,
and presentation-resource interpretation. A Source 1 backend must never load a
Source 2 replacement asset or silently render a bind pose. Missing exact assets
or insufficient animation inputs produce a structured refusal.

Both sides exchange the existing canonical entity shape:

```json
{
  "id": "source1:edict:17",
  "generation": 4,
  "class": "CCSPlayer",
  "components": {
    "transform": {
      "provenance": "recorded",
      "value": {"origin": [1.0, 2.0, 3.0]}
    },
    "source1_reconstruction": {
      "provenance": "derived",
      "derivation": "coverage classification from decoded demo fields",
      "value": {
        "animation": {
          "status": "unavailable",
          "reason": "final client-evaluated bones are absent from this demo"
        }
      }
    }
  }
}
```

An edict serial change advances the canonical generation. Leaving the PVS is
not automatically equivalent to authoritative destruction unless the demo
update says it is deleted; the normalizer must retain that distinction.

Decoded global lookup state is canonical state too. String tables are emitted
as hashed resource entities, not retained solely in parser memory, so a seeked
or reopened checkpoint contains the same model/user/material lookup generation
that produced an edict binding. A `source1_model` path is a lossless derived
join between recorded `m_nModelIndex` and the corresponding recorded,
canonical `modelprecache` entry; it is not mislabeled as a directly recorded
path.

## Source 1 evidence levels

Implementation and reports distinguish these levels:

1. demo header and command framing;
2. packet/net-message framing;
3. send-table, class, string-table, and baseline reconstruction;
4. packet-entity create/update/leave/delete reduction;
5. game-event and ego-action alignment;
6. canonical component normalization;
7. exact Source 1 asset decoding;
8. exact animation-state evaluation and rendering.

Passing a lower level is not evidence that a higher level works. Synthetic
fixtures establish corruption handling and deterministic transitions, but do
not establish compatibility with a real legacy CS:GO recording. That requires
an immutable real `.dem`, exact build/content provenance, and an end-to-end
coverage report.

## End-to-end acceptance

A variant is selectable only when a profile resolves to its matching backend
without fallback. A state-reconstruction demonstration additionally requires:

- a complete initial checkpoint and ordered transactions;
- deterministic seek equivalence and state hashes;
- explicit unknown/unavailable state rather than invented defaults;
- a renderer output joined to the exact input state hash;
- a gap manifest for every canonical field the backend did not consume; and
- independent evidence for parser correctness, state correctness, and image
  production.

The deterministic Source 1 reference renderer is useful for exercising this
contract before all proprietary game assets are available. It is not a claim
of visual parity with CS:GO and must identify itself as a reference backend in
its output manifest.

The implemented reference path verifies the canonical entity hash before
rendering, parses VBSP world faces and Source 1 MDL 48/49 + VVD 4 + DX90.VTX 7
LOD0 geometry, consumes only explicit final bone matrices for skinning, and
emits independently hash-verified P6 color plus uint64 depth artifacts. It
still refuses authoritative parity, animation evaluation from sequence state,
materials/lightmaps, flexes, particles, and HUD reconstruction.
