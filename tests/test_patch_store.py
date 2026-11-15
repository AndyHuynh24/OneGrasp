from types import SimpleNamespace

import numpy as np

from touch.data.patch_store import build_store


def _seq(name):
    return SimpleNamespace(source="src", participant="p_001", name=name, views=["overhead"])


def _write(root, s, jpeg, offset, length):
    d = root / s.source / s.participant / s.name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "overhead.npz", jpeg=jpeg, offset=offset, length=length)


def _compact(root, s):  # what scripts/extract_patches.py does after packing
    p = root / s.source / s.participant / s.name / "overhead.npz"
    with np.load(p) as z:
        d = {k: z[k] for k in z.files if k != "jpeg"}
    np.savez(p, **d)


def _patch(root, key, i, f):
    with np.load(root / "src_index.npz") as z:
        starts = np.concatenate([[0], np.cumsum(z["sizes"])])
        k = list(z["keys"]).index(key)
        off, ln = z["offset"][starts[k] + i, f], z["length"][starts[k] + i, f]
    return (root / "src_store.bin").read_bytes()[off : off + ln]


def test_build_store_appends_and_keeps_packed_sequences(tmp_path):
    a, b = _seq("a"), _seq("b")
    off_a = np.array([[0, 3, -1, -1, -1], [5, -1, -1, -1, -1]], np.int64)
    len_a = np.array([[3, 2, 0, 0, 0], [4, 0, 0, 0, 0]], np.int32)
    _write(tmp_path, a, np.frombuffer(b"AAABBCCCC", np.uint8), off_a, len_a)
    build_store("src", [a], tmp_path)
    _compact(tmp_path, a)

    off_b = np.array([[0, -1, -1, -1, -1]], np.int64)
    len_b = np.array([[2, 0, 0, 0, 0]], np.int32)
    _write(tmp_path, b, np.frombuffer(b"ZZ", np.uint8), off_b, len_b)
    build_store("src", [a, b], tmp_path)  # a has no jpeg any more: must be kept, not dropped

    with np.load(tmp_path / "src_index.npz") as z:
        assert list(z["keys"]) == ["p_001/a/overhead", "p_001/b/overhead"]
        assert z["offset"].shape == (3, 5) and z["length"].shape == (3, 5)
        assert (z["offset"][:2] == off_a).all()  # old offsets unchanged
        assert z["offset"][2, 0] == 9 and (z["offset"][2, 1:] == -1).all()
    assert _patch(tmp_path, "p_001/a/overhead", 0, 1) == b"BB"
    assert _patch(tmp_path, "p_001/a/overhead", 1, 0) == b"CCCC"
    assert _patch(tmp_path, "p_001/b/overhead", 0, 0) == b"ZZ"
    assert not (tmp_path / "src_store.tmp").exists()
