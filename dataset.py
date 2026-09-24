"""Finding identities in whatever folder the user points at.

Expected layout - one sub-folder per identity, with one or more reference
images and optional probe/match images:

    <root>/
        <identity>/reference_frontal.jpg  enrolled into the 1:N gallery
        <identity>/reference_left30.jpg   enrolled into the 1:N gallery
        <identity>/match_01.jpg           used as a probe

The same files may be placed in `<identity>/references/` and
`<identity>/matches/` sub-folders.

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
    __slots__ = ("name", "references", "matches")

    def __init__(self, name, references, matches):
        self.name = name
        self.references = references
        self.matches = matches

    @property
    def reference(self):
        """Backward-compatible access to the first reference image."""
        return self.references[0] if self.references else None

    @property
    def match(self):
        """Backward-compatible access to the first match image."""
        return self.matches[0] if self.matches else None


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
    """Images whose filename is `<stem>` or starts with `<stem>_`."""
    prefix = stem.lower() + "_"
    return [
        os.path.join(folder, f)
        for f in _images(folder)
        if (os.path.splitext(f)[0].lower() == stem.lower()
            or os.path.splitext(f)[0].lower().startswith(prefix))
    ]


def _identity_images(folder):
    """Return reference and match images from flat or split-folder layouts."""
    references_dir = os.path.join(folder, "references")
    matches_dir = os.path.join(folder, "matches")
    if os.path.isdir(references_dir) or os.path.isdir(matches_dir):
        references = [os.path.join(references_dir, f)
                      for f in _images(references_dir)]
        matches = [os.path.join(matches_dir, f)
                   for f in _images(matches_dir)]
        return references, matches

    images = [os.path.join(folder, f) for f in _images(folder)]
    references = _named(folder, "reference")
    matches = _named(folder, "match")
    if references or matches:
        return references, matches
    return images[:1], images[1:]


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
    root = os.path.abspath(root)
    files = []
    for folder, _dirs, names in os.walk(root):
        for name in sorted(names, key=_natural):
            path = os.path.join(folder, name)
            if name.lower().endswith(IMAGE_EXTS):
                stat = os.stat(path)
                files.append((os.path.relpath(path, root), stat.st_size,
                              stat.st_mtime_ns))
    key = repr((os.path.normcase(root), files)).encode("utf-8", "replace")
    return hashlib.sha1(key).hexdigest()[:10]


def scan(root, limit=0):
    """Inspect a dataset root. Returns a dict; never raises on odd folders."""
    root = resolve_root(root)
    result = {
        "root": root, "identities": [], "convention": None,
        "n_with_match": 0, "n_references": 0, "skipped": 0, "warnings": [],
    }
    if not os.path.isdir(root):
        result["warnings"].append(f"Not a folder: {root}")
        return result

    folders = _subdirs(root)
    if not folders:
        result["warnings"].append(
            "No sub-folders found. Expected one folder per identity, each "
            "containing one or more reference images.")
        return result

    # Decide the convention from a sample rather than per-folder, so a dataset is
    # read one consistent way instead of a mix.
    sample = folders[:50]
    named = sum(1 for d in sample
                if _identity_images(os.path.join(root, d))[0])
    result["convention"] = "reference/match" if named > len(sample) // 2 else "positional"

    for name in folders:
        folder = os.path.join(root, name)
        references, matches = _identity_images(folder)

        if not references:
            result["skipped"] += 1
            continue
        result["identities"].append(Identity(name, references, matches))
        result["n_references"] += len(references)
        result["n_with_match"] += len(matches)
        if limit and len(result["identities"]) >= limit:
            break

    if not result["identities"]:
        result["warnings"].append(
            f"Found {len(folders)} sub-folders but no usable images in them "
            f"(looked for {', '.join(IMAGE_EXTS)}).")
    elif result["convention"] == "positional":
        result["warnings"].append(
            "No reference*/match* filenames found - falling back to "
            "alphabetical order: first image enrolled, remaining images used as probes.")
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
    return (f"{n:,} identities - {result['n_references']:,} reference images, "
            f"{result['n_with_match']:,} match images ({result['convention']} layout)")


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
