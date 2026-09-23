"""Finding identities in whatever folder the user points at.

Expected layout - one sub-folder per identity, two images inside:

    <root>/
        <identity>/reference.jpg     enrolled into the 1:N gallery
        <identity>/match.jpg         used as the probe / as the 1:1 pair

A `.zip` of that folder is accepted too and extracted next to the app. If the
archive wraps everything in a single top-level directory, we descend into it, so
users do not have to care how the zip was made.

Datasets that only have folder-per-identity with arbitrary filenames still work:
the first image is enrolled and the second becomes the probe. `scan()` reports
which convention it used so a wrong folder is obvious before enrolling.
"""
import hashlib
import os
import re
import zipfile

from config import IMAGE_EXTS


class Identity:
    __slots__ = ("name", "reference", "match")

    def __init__(self, name, reference, match):
        self.name = name
        self.reference = reference
        self.match = match


def _natural(name):
    """Sort '2' before '10' - keeps identity ordering stable across machines."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _images(folder):
    try:
        entries = os.listdir(folder)
    except OSError:
        return []
    return sorted((f for f in entries if f.lower().endswith(IMAGE_EXTS)), key=_natural)


def _named(folder, stem):
    """First image called `<stem>.<ext>` in this folder."""
    for f in _images(folder):
        if os.path.splitext(f)[0].lower() == stem:
            return os.path.join(folder, f)
    return None


def _subdirs(root):
    try:
        entries = os.listdir(root)
    except OSError:
        return []
    return sorted((d for d in entries if os.path.isdir(os.path.join(root, d))),
                  key=_natural)


def resolve_root(root):
    """Descend through wrapper directories created by zipping a folder.

    A zip of `final_dataset` usually extracts to `<dest>/final_dataset/...`; this
    walks down as long as there is exactly one sub-directory and no images at the
    current level.
    """
    root = os.path.abspath(root)
    for _ in range(4):
        subs = _subdirs(root)
        if len(subs) == 1 and not _images(root):
            nested = os.path.join(root, subs[0])
            # Stop if the single sub-directory is itself an identity folder.
            if _subdirs(nested) or not _images(nested):
                root = nested
                continue
        break
    return root


def dataset_id(root):
    """Short stable id for a dataset location, used in the gallery filename."""
    key = os.path.normcase(os.path.abspath(root)).encode("utf-8", "replace")
    return hashlib.sha1(key).hexdigest()[:10]


def scan(root, limit=0):
    """Inspect a dataset root. Returns a dict; never raises on odd folders."""
    root = resolve_root(root)
    result = {
        "root": root, "identities": [], "convention": None,
        "n_with_match": 0, "skipped": 0, "warnings": [],
    }
    if not os.path.isdir(root):
        result["warnings"].append(f"Not a folder: {root}")
        return result

    folders = _subdirs(root)
    if not folders:
        result["warnings"].append(
            "No sub-folders found. Expected one folder per identity, each "
            "containing a reference and a match image.")
        return result

    # Decide the convention from a sample rather than per-folder, so a dataset is
    # read one consistent way instead of a mix.
    sample = folders[:50]
    named = sum(1 for d in sample if _named(os.path.join(root, d), "reference"))
    result["convention"] = "reference/match" if named > len(sample) // 2 else "positional"

    for name in folders:
        folder = os.path.join(root, name)
        if result["convention"] == "reference/match":
            reference = _named(folder, "reference")
            match = _named(folder, "match")
        else:
            imgs = [os.path.join(folder, f) for f in _images(folder)]
            reference = imgs[0] if imgs else None
            match = imgs[1] if len(imgs) > 1 else None

        if reference is None:
            result["skipped"] += 1
            continue
        result["identities"].append(Identity(name, reference, match))
        if match:
            result["n_with_match"] += 1
        if limit and len(result["identities"]) >= limit:
            break

    if not result["identities"]:
        result["warnings"].append(
            f"Found {len(folders)} sub-folders but no usable images in them "
            f"(looked for {', '.join(IMAGE_EXTS)}).")
    elif result["convention"] == "positional":
        result["warnings"].append(
            "No reference.*/match.* filenames found - falling back to "
            "alphabetical order: first image enrolled, second used as probe.")
    if result["identities"] and result["n_with_match"] == 0:
        result["warnings"].append(
            "No match images found. 1:N still works with uploaded probes, but "
            "bench.py has no probe/pair source.")
    if result["skipped"]:
        result["warnings"].append(f"Skipped {result['skipped']} folder(s) with no image.")
    return result


def summary(result):
    """One-line human description of a scan, for the setup screen."""
    n = len(result["identities"])
    if not n:
        return "no identities found"
    return (f"{n:,} identities - {result['n_with_match']:,} with a match image "
            f"({result['convention']} layout)")


def extract_zip(zip_path, dest_dir):
    """Extract a dataset archive and return the folder that holds the identities."""
    zip_path = os.path.abspath(zip_path)
    name = os.path.splitext(os.path.basename(zip_path))[0]
    target = os.path.join(dest_dir, name)
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            # Refuse absolute or parent-escaping members (zip-slip).
            out = os.path.normpath(os.path.join(target, member.filename))
            if not out.startswith(os.path.normpath(target) + os.sep) and out != os.path.normpath(target):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(target)
    return resolve_root(target)
