"""Tests for scripts/organise_raw.py."""

from pathlib import Path

from scripts.organise_raw import organise


def test_flat_layout_gets_organised(make_dataset):
    root: Path = make_dataset(n_pairs=4, layout="flat")
    rc = organise(root, root, copy=False)
    assert rc == 0
    assert (root / "images").is_dir()
    assert (root / "labels").is_dir()
    assert len(list((root / "images").glob("*.jpg"))) == 4
    assert len(list((root / "labels").glob("*.txt"))) == 4


def test_already_organised_is_noop(make_dataset):
    root: Path = make_dataset(n_pairs=4, layout="separated")
    rc = organise(root, root, copy=False)
    assert rc == 0
    # Files stay where they were; counts unchanged.
    assert len(list((root / "images").glob("*.jpg"))) == 4
    assert len(list((root / "labels").glob("*.txt"))) == 4


def test_orphan_image_reported_but_not_fatal(make_dataset, capsys):
    root: Path = make_dataset(n_pairs=2, layout="flat")
    (root / "lonely.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # image with no label
    rc = organise(root, root, copy=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert "image(s) without a label" in out


def test_meta_files_ignored(make_dataset):
    root: Path = make_dataset(
        n_pairs=2,
        layout="flat",
        extra_files=[("classes.txt", "bowl_empty\nbowl_not_empty\n")],
    )
    rc = organise(root, root, copy=False)
    assert rc == 0
    # classes.txt must NOT have been moved into labels/
    assert (root / "classes.txt").is_file()
    assert not (root / "labels" / "classes.txt").exists()


def test_copy_mode_preserves_source(make_dataset):
    root: Path = make_dataset(n_pairs=3, layout="flat")
    out = root.parent / "out"
    rc = organise(root, out, copy=True)
    assert rc == 0
    # Source still has originals
    assert len(list(root.glob("*.jpg"))) == 3
    assert len(list(root.glob("*.txt"))) == 3
    # Destination has copies
    assert len(list((out / "images").glob("*.jpg"))) == 3
    assert len(list((out / "labels").glob("*.txt"))) == 3
