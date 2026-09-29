# Pinned third-party reference source

Unmodified Microsoft Vidur source and its MIT license, commit `abae7f63aa857300f5cdc6f5e0d27860cd24721b`. This file is upstream code, not an original contribution of this repository.

The artifact never imports this module. `audit_vidur.py` verifies its Git blob hash and extracts only the two reviewed prefill-feature methods. A key-capturing lookup records their actual input keys; no trained latency model or whole Vidur simulator is evaluated.

Source: https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/vidur/execution_time_predictor/sklearn_execution_time_predictor.py
