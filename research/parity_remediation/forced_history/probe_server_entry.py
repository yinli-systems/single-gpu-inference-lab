"""Uninstrumented SGLang entrypoint for custom-processor position semantics."""
import runpy

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
