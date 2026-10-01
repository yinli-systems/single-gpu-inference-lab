"""Private transform anchors, legal arms and exact mangled-name normalization."""

import pytest

from research.selector_v4.exposed_diagnostics.attribute_launch_control import (
    ARMS,
    normalize_symbols,
    rewrite,
)


def source():
    return "\n".join(
        f"void BatchPrefillWith{layout}KVCacheResourceKernel() {{ immutable_body(); }}\n"
        "smem_size = 65536;\n"
        "cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);"
        for layout in ("Ragged", "Paged")
    )


@pytest.mark.parametrize("attribute,launch", ARMS)
def test_only_optin_launch_and_symbol_identity_change(attribute, launch):
    original = source()
    changed = rewrite(original, attribute, launch)
    for layout in ("Ragged", "Paged"):
        prefix = f"BatchPrefillWith{layout}KVCacheResource"
        changed = changed.replace(prefix + f"A{attribute}L{launch}", prefix)
    changed = changed.replace(
        f"smem_size = {launch}; const int diagnostic_optin_bytes = {attribute};",
        "smem_size = 65536;",
    ).replace(
        "cudaFuncAttributeMaxDynamicSharedMemorySize, diagnostic_optin_bytes",
        "cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size",
    )
    assert changed == original


def test_unreviewed_anchor_or_illegal_launch_above_attribute_is_rejected():
    with pytest.raises(ValueError, match="legal arms"):
        rewrite(source(), 49152, 65536)
    with pytest.raises(ValueError, match="anchors"):
        rewrite(source().replace("smem_size = 65536;", "changed", 1), 65536, 49152)


@pytest.mark.parametrize("attribute,launch", ARMS)
def test_normalization_changes_only_exact_identifier_and_length(attribute, launch):
    for layout in ("Paged", "Ragged"):
        arm = f"BatchPrefillWith{layout}KVCacheResourceA{attribute}L{launch}Kernel"
        native = f"BatchPrefillWith{layout}KVCacheKernel"
        raw = f"Function : _Z{len(arm)}{arm}OtherTypes\nMOV R1, 0x100;\n"
        expected = f"Function : _Z{len(native)}{native}OtherTypes\nMOV R1, 0x100;\n"
        assert normalize_symbols(raw, attribute, launch) == expected
        assert normalize_symbols(raw, 123, 456) == raw
