"""Build and verify the frozen benchmark inputs and read-only RL4CO source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np


FIXED_DIR = Path(__file__).resolve().parent / "fixed"
PROJECT_ROOT = FIXED_DIR.parents[2]
SOURCE_TREE = "194dc6ae564df2c836c8f3da1ce2496ed983c477"
SOURCE_COMMIT = "1bfbb3415e997ee042e46ff286fa33adaa04bd17"

# Separate seeds keep each experiment population fixed when another set is added.
INSTANCE_COUNT = 1024
SET_SPECS = {
    "tsp50": ("tsp", 50, 2026111950),
    "tsp100": ("tsp", 100, 2026111900),
    "tsp200": ("tsp", 200, 2026111902),
    "cvrp100": ("cvrp", 100, 2026111910),
}
SOURCE_SUFFIXES = {
    ".py", ".pyc", ".pyi", ".pyd", ".so", ".dll", ".dylib", ".pth", ".yaml", ".yml",
}
IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _git_object_hash(kind: bytes, content: bytes) -> bytes:
    prefix = kind + b" " + str(len(content)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + content, usedforsecurity=False).digest()


def _tree_hash(files: dict) -> bytes:
    # Git sorts a tree as though directory names end in a slash.
    entries = sorted(files, key=lambda key: key + b"/" if isinstance(files[key], dict) else key)
    parts = []
    for key in entries:
        value = files[key]
        mode, digest = (b"40000", _tree_hash(value)) if isinstance(value, dict) else value
        parts.append(mode + b" " + key + b"\0" + digest)
    return _git_object_hash(b"tree", b"".join(parts))


def verify_rl4co(root: Path | str = PROJECT_ROOT) -> dict:
    """Verify actual vendored file bytes and reject unexpected runtime source files.

    Use the outer project's Git index for names and Git modes. Reconstruct its
    vendor tree from disk so the caller can run from any project commit retaining
    the same vendor snapshot. Windows CRLF is accepted if the canonical blob is LF.
    """
    root = Path(root).resolve()
    vendor = root / "rl4co"
    if vendor.is_symlink() or not vendor.is_dir():
        raise ValueError(f"Expected the vendored RL4CO directory at {vendor}")
    tracked = set()
    tree: dict = {}
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "--full-name", "-z", "--", "rl4co/"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=60,
    ).stdout
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        index_fields, full_name = entry.split(b"\t", 1)
        mode, expected, stage = index_fields.split()
        if stage != b"0" or mode not in {b"100644", b"100755"}:
            raise ValueError(f"Unmerged or unsupported Git index entry: {os.fsdecode(full_name)}")
        top, *parts = full_name.split(b"/")
        if top != b"rl4co" or not parts:
            raise ValueError(f"Unexpected vendor path: {os.fsdecode(full_name)}")
        path = root / os.fsdecode(full_name)
        relative_parents = path.relative_to(vendor).parents
        if (
            path.is_symlink()
            or not path.is_file()
            or any((vendor / parent).is_symlink() for parent in relative_parents)
        ):
            raise ValueError(f"Missing, linked or non-regular vendor file: {path}")
        payload = path.read_bytes()
        digest = _git_object_hash(b"blob", payload)
        if digest.hex().encode("ascii") != expected and b"\r\n" in payload:
            digest = _git_object_hash(b"blob", payload.replace(b"\r\n", b"\n"))
        if digest.hex().encode("ascii") != expected:
            raise ValueError(f"Vendor file differs from the Git index: {path}")
        subtree = tree
        for part in parts[:-1]:
            subtree = subtree.setdefault(part, {})
        subtree[parts[-1]] = (mode, digest)
        tracked.add(path)
    actual = _tree_hash(tree).hex()
    if actual != SOURCE_TREE:
        raise ValueError(f"RL4CO source tree mismatch: expected {SOURCE_TREE}, found {actual}")
    # configs/env/ exists in the original local installation but was ignored
    # by the vendored .gitignore. It is unused by the direct T2 Python runner.
    overlay = json.loads(
        (FIXED_DIR.parent / "local-hydra-overlay.json").read_text(encoding="utf-8")
    )
    seen_overlay = set()
    for directory, subdirs, filenames in os.walk(vendor):
        here = Path(directory)
        subdirs[:] = sorted(set(subdirs) - IGNORED_DIRS)
        runtime_dir = here.relative_to(vendor).parts[:1] in {("rl4co",), ("configs",)}
        for name in sorted([*filenames, *(d for d in subdirs if (here / d).is_symlink())]):
            path = here / name
            if path in tracked or name == ".DS_Store":
                continue
            relpath = path.relative_to(vendor).as_posix()
            if relpath in overlay and path.is_file() and not path.is_symlink():
                digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                if digest != overlay[relpath]:
                    raise ValueError(f"Local Hydra overlay file differs from its pin: {path}")
                seen_overlay.add(relpath)
            elif runtime_dir or path.suffix.lower() in SOURCE_SUFFIXES or path.is_symlink():
                raise ValueError(f"Untracked file inside the vendored source: {path}")
    if seen_overlay and seen_overlay != set(overlay):
        missing = sorted(set(overlay) - seen_overlay)
        raise ValueError(f"Local Hydra overlay is incomplete; missing: {', '.join(missing)}")
    return {
        "path": str(vendor),
        "git_tree": SOURCE_TREE,
        "pinned_in_project_commit": SOURCE_COMMIT,
        "verified_files": len(tracked),
        "verified_local_hydra_overlay_files": len(seen_overlay),
    }


def _canonical_manifest() -> dict:
    return json.loads((FIXED_DIR / "manifest.json").read_text(encoding="utf-8"))


def load_instance_set(name: str, directory: Path | str = FIXED_DIR) -> tuple[dict, dict]:
    """Read one fixed set and verify the exact NPZ bytes against the frozen manifest."""
    entry = _canonical_manifest()["sets"][name]
    path = Path(directory) / entry["file"]
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        while chunk := stream.read(1 << 20):
            checksum.update(chunk)
        if checksum.hexdigest() != entry["sha256"]:
            raise ValueError(f"Fixed instance checksum mismatch: {path}")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    _validate_arrays(arrays, entry)
    return arrays, entry


def as_rl4co_tensordict(arrays: dict, device: str = "cpu"):
    """Convert a fixed batch to RL4CO's input representation for env.reset()."""
    import torch
    from tensordict import TensorDict

    values = {"locs": np.asarray(arrays["locs"], dtype=np.float32)}
    if "depot" in arrays:
        capacity = np.asarray(arrays["capacity"], dtype=np.float32)
        values.update(
            depot=np.asarray(arrays["depot"], dtype=np.float32),
            demand=np.asarray(arrays["demand"], dtype=np.float32) / capacity,
            capacity=capacity,
        )
    return TensorDict(
        {key: torch.from_numpy(value).to(device) for key, value in values.items()},
        batch_size=[len(values["locs"])],
    )


def _validate_arrays(arrays: dict, entry: dict) -> None:
    count, size = entry["count"], entry["size"]
    spec = {"locs": ((count, size, 2), np.dtype("<f4"))}
    if entry["problem"] == "cvrp":
        spec.update(
            depot=((count, 2), np.dtype("<f4")),
            demand=((count, size), np.dtype("<i8")),
            capacity=((count, 1), np.dtype("<i8")),
        )
    if set(arrays) != set(spec):
        raise ValueError(f"Unexpected instance fields in {entry['file']}")
    for key, (shape, dtype) in spec.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} in {entry['file']}: expected {shape} of {dtype}")
    for key in {"locs", "depot"} & set(arrays):
        if not ((arrays[key] >= 0) & (arrays[key] < 1)).all():
            raise ValueError(f"Expected unit-square coordinates in {entry['file']}")
    if entry["problem"] == "cvrp":
        if not ((arrays["demand"] >= 1) & (arrays["demand"] <= 9)).all():
            raise ValueError(f"Invalid integer customer demand in {entry['file']}")
        if not (arrays["capacity"] == 50).all():
            raise ValueError(f"Invalid CVRP100 vehicle capacity in {entry['file']}")


def _make_arrays(problem: str, size: int, count: int, seed: int) -> dict:
    rng = np.random.PCG64(seed)

    def coordinates(shape: tuple) -> np.ndarray:
        words = rng.random_raw(shape)
        return ((words >> 40).astype("<f4") * np.float32(2**-24)).astype("<f4")

    arrays = {"locs": coordinates((count, size, 2))}
    if problem == "cvrp":
        arrays["depot"] = coordinates((count, 2))
        # Rejection avoids modulo bias. Only one 64-bit word out of 2^64 is rejected.
        words = rng.random_raw((count, size))
        rejected = words == np.iinfo(np.uint64).max
        while rejected.any():
            words[rejected] = rng.random_raw(int(rejected.sum()))
            rejected = words == np.iinfo(np.uint64).max
        arrays["demand"] = ((words % 9) + 1).astype("<i8")
        arrays["capacity"] = np.full((count, 1), 50, dtype="<i8")
    return arrays


def _write_set(path: Path, arrays: dict) -> str:
    # Fix archive metadata and use stored entries so creation OS and zlib
    # versions do not affect the archived bytes.
    with path.open("xb") as stream, zipfile.ZipFile(stream, mode="w") as archive:
        for key, array in arrays.items():
            info = zipfile.ZipInfo(key + ".npy")
            info.create_system = 3
            info.external_attr = 0o600 << 16
            with archive.open(info, "w") as member:
                np.lib.format.write_array(member, array, version=(1, 0), allow_pickle=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [*(directory / f"{name}.npz" for name in SET_SPECS), directory / "manifest.json"]
    if any(path.exists() for path in outputs):
        raise ValueError(f"Refusing to overwrite fixed datasets in {directory}")
    sets = {}
    for name, (problem, size, seed) in SET_SPECS.items():
        file = f"{name}.npz"
        arrays = _make_arrays(problem, size, INSTANCE_COUNT, seed)
        entry = {"file": file, "problem": problem, "size": size, "count": INSTANCE_COUNT, "seed": seed}
        _validate_arrays(arrays, entry)
        entry["sha256"] = _write_set(directory / file, arrays)
        sets[name] = entry
    manifest = {
        "schema_version": 1,
        "generation": "PCG64; uniform 24-bit unit-square coordinates; CVRP100 integer demand 1–9",
        "sets": sets,
    }
    with (directory / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("--directory", type=Path, default=FIXED_DIR)
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.directory)
        print(f"Generated fixed instances in {args.directory}")
    else:
        provenance = verify_rl4co()
        sets = {
            name: load_instance_set(name, args.directory)[1]
            for name in _canonical_manifest()["sets"]
        }
        print(json.dumps({"rl4co": provenance, "sets": sets}, indent=2))


if __name__ == "__main__":
    main()
