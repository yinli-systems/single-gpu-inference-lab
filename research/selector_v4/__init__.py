"""Safe deployment-matched resource-tactic autotuning research prototype.

Modules are intentionally not imported here: manifest and cache tooling must remain
usable in a Python standard-library-only environment, while the statistical tuner
uses NumPy in its explicit module.
"""

__version__ = "4.0.0"
