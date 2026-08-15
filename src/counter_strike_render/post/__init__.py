"""W5: the passes between opaque shading and the final raster.

Every module here is IMPORTABLE — it parses no argv, reads no globals, and
touches no filesystem at import. That is a hard requirement, not a style
preference: `gpu_render.py` calls `argparse.parse_args()` at module scope,
so nothing defined inside it can be imported by a test or by the
conformance registry. An expression no test can reach is an expression
whose conformance is an assertion.

`gpu_render.py` calls into these modules; the conformance registry resolves
the same callables by real import. One implementation, two callers.
"""
