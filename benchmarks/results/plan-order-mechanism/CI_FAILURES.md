# Retained CI failure

Initial integration commit e128bce5d363e5fa6c8b1ad71ed541afa6b8646e failed CI run 36571218491 during collection/execution of the artifact native test driver, with NameError: __file__. No GPU benchmark was executed by that CI run. Main pytest collection is now scoped to tests/, while artifact contract suites and both native drivers have explicit subprocess CI steps. The test runner accepts --out NEW_PATH and never overwrites the recorded qualification by default. This changes test invocation/packaging, not frozen GPU code, endpoints or measurements. The final pushed commit needs its own green CI before claiming CI success.

The final local documentation-only SSH push encountered Permission denied (publickey). No key, access setting or shared runtime was modified. These documentation corrections are published through the already-authorized GitHub connector; the prior complete evidence commit remains in history.
