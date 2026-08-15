"""Importable shading modules the renderer CALLS.

`gpu_render.py` parses argv at import, so nothing defined inside it can be
imported by a test, a registry, or another module. Every expression in
there is therefore unreachable by the conformance runner by construction
-- and, as `conformance/terms/_TEMPLATE.py` puts it, "an expression no
test can reach is an expression whose conformance is an assertion."

This package is the other side of that. Modules here:

  * import nothing from `gpu_render.py` (no argv, no globals, no device);
  * take every input as an argument, including the constants;
  * work on EITHER numpy arrays or torch tensors, so the renderer calls
    them on the GPU and the conformance runner calls them on numpy
    without a second copy existing.

The dual-backend rule is what keeps the registry honest. If the runner
had to call a numpy re-implementation of a torch function, the thing
under test would be the re-implementation. Here it is the same code
object the renderer runs.
"""
