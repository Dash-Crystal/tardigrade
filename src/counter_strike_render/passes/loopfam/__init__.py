"""Renderer-side implementations of CS2's loop-carrying pixel families.

One module per functional group. Everything here is IMPORTABLE and free of
argv parsing: the conformance registry resolves `impl=` against these
modules with a real import, and an expression no test can reach is an
expression whose conformance is an assertion.

Every function takes and returns numpy arrays with a leading batch axis, so
one call covers a whole tile and the registry can hand it 4096 synthetic
samples without a shim.

WHAT THE A/B HERE CAN AND CANNOT CATCH. The reference side of each pair is
a transcription of the decompiled GLSL, written by the same reader who
wrote this side. A vectorisation defect -- wrong `where` operand, a
reduction on the wrong axis, a clamp on the wrong side of a divide -- shows
up as a delta. A MISREADING of the GLSL does not: it is wrong identically
on both sides. That ceiling is the registry's declared lane-2 gap, not a
property of these modules, and it is why the decompiled source and its
sha256 are committed next to every term.
"""
