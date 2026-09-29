# Validation history

This implementation and review were AI-assisted. CPU reruns on the remote Linux environment are not third-party reproduction.

- Initial Mac system Python had no NumPy; its pipeline tests could not run there. No shared Python environment was changed.
- An encoded source-transfer attempt failed decompression before writing its target. Chunk hashes and the final file hash were subsequently checked before writing.
- A new diagnostic test initially lacked its fixture parent directory; this synthetic test setup was repaired before the 75-test passing run.
- The real nvidia-smi CSV header includes `power.limit [W]`; the new adapter now handles that exact header and has a regression test. Original GPU source/data were not altered.
- Two coverage-instrumented local test attempts timed out. No final coverage percentage is claimed. Uninstrumented local 75 tests and Linux 75 plus 31 regression tests passed.
- HTTP parity failure remains in the original files and the published gate/diagnostic receipts. No failing request or measured block was dropped.
- Source, protocol, driver and profiling limitations remain explicit. No production default, service speedup or new-to-literature guarantee is asserted.
