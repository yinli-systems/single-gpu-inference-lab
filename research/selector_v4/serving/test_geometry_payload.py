"""CPU checks of actual native page layouts, including SGLang's implicit page1."""

import pytest

torch = pytest.importorskip("torch")

from research.selector_v4.serving.geometry_server import unpack_paged_payload


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("page_size,implicit", [(1, True), (1, False), (16, False)])
@pytest.mark.parametrize("packed", [False, True])
def test_actual_page_dimension_and_kv_views(layout, page_size, implicit, packed):
    shape = (5, page_size, 8, 128) if layout == "NHD" else (5, 8, page_size, 128)
    k = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    v = k + 500000
    if implicit:
        k, v = (x.squeeze(1 if layout == "NHD" else 2) for x in (k, v))
    cache = torch.stack((k, v), dim=1) if packed else (k, v)
    actual_k, actual_v, actual_page = unpack_paged_payload(cache, layout)
    assert actual_page == page_size and actual_k.shape == actual_v.shape == shape
    assert actual_k.reshape(-1).tolist() == k.reshape(-1).tolist()
    assert actual_v.reshape(-1).tolist() == v.reshape(-1).tolist()
    # A page1 prefix with two/three pages is two/three tokens, not 9/17.
    indptr, tails = [0, 2, 5], [1, page_size]
    lengths = [(b - a - 1) * actual_page + n for a, b, n in zip(indptr, indptr[1:], tails)]
    assert lengths == [page_size + 1, 3 * page_size]
    if not packed:
        assert actual_k.data_ptr() == k.data_ptr() and actual_v.data_ptr() == v.data_ptr()
