#!/usr/bin/env python3
"""Populate a data folder from the Mendeley Data deposit that accompanies this code.

The deposit is public, so no account, token or API key is needed:

    python fetch_data.py --dest ./data              # tables + the 40 validation scans (~1.1 GB)
    python fetch_data.py --dest ./data --what tables   # the two tables only (~35 MB)
    python fetch_data.py --dest ./data --what all      # adds the 322 field scans (~6.0 GB)

The notebook needs `--what validation`, which is the default: sections 1 to 3 recompute
features from the 40 annotated scans, and section 4 reads the deposited detection table
rather than re-segmenting the full survey.

Files already present with the right SHA-256 are left alone, so the script is resumable.
"""

import argparse
import hashlib
import sys
from pathlib import Path

try:
    import requests
except ImportError:                                    # pragma: no cover
    sys.exit("fetch_data.py needs the `requests` package:  pip install requests")

DOI = "10.17632/6pbgs93kdf"
DATASET_ID = "6pbgs93kdf"
RECORD_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"

# The deposit state this release of the code was built and verified against. The anonymous
# API only lists the files of the *current* version, so these are how you tell whether the
# deposit has moved on since the release. A mismatch is a warning, not an error.
EXPECTED_VERSION = 2
EXPECTED = {
    "insects.csv":
        "42a4962ae9cfd7367dd55e6a773be4c21b2cfac07c9db98de9f2034d3e8a4d16",
    "field_traps.csv":
        "3b3a81aacabc5423272006ef055a7429b66c6959741437cd7d3674cf76a54298",
}

TABLES = ["insects.csv", "field_traps.csv"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def record():
    response = requests.get(RECORD_URL, timeout=120)
    response.raise_for_status()
    return response.json()


def folders(rec):
    """Identify the two image folders by the table each one contains.

    Folder ids are not stable across deposit versions and the anonymous record carries no
    folder names, so the tables are the landmarks.
    """
    where = {f["filename"]: f["folder_id"] for f in rec["files"]}
    missing = [t for t in TABLES if t not in where]
    if missing:
        sys.exit(f"the deposit does not contain {', '.join(missing)} - it may be an older "
                 f"version than this code expects (version {EXPECTED_VERSION})")
    return {"validation": where["insects.csv"], "field": where["field_traps.csv"]}


def download(entry, dest):
    details = entry["content_details"]
    expected = details.get("sha256_hash")
    if dest.exists() and expected and sha256(dest) == expected:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(details["download_url"], stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
    if expected and sha256(tmp) != expected:
        tmp.unlink()
        sys.exit(f"{dest.name}: SHA-256 does not match the deposit listing")
    tmp.replace(dest)
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, required=True,
                        help="data folder to populate; use it as GPSVDI_DATA_ROOT")
    parser.add_argument("--what", choices=["tables", "validation", "all"],
                        default="validation",
                        help="tables only, tables + the 40 annotated scans (default), "
                             "or everything including the 322 field scans")
    args = parser.parse_args()

    rec = record()
    print(f"deposit {DOI}  version {rec['version']}  ({len(rec['files'])} files)")
    if rec["version"] != EXPECTED_VERSION:
        print(f"  note: this code was verified against version {EXPECTED_VERSION}. "
              f"See the release notes for what changed.")

    ids = folders(rec)
    wanted = []
    for entry in rec["files"]:
        name, folder = entry["filename"], entry["folder_id"]
        if name in TABLES:
            wanted.append((entry, args.dest / name))
        elif args.what in ("validation", "all") and folder == ids["validation"]:
            wanted.append((entry, args.dest / "40traps" / name))
        elif args.what == "all" and folder == ids["field"]:
            wanted.append((entry, args.dest / "field_traps" / name))

    total = sum(int(e["size"]) for e, _ in wanted)
    print(f"{len(wanted)} files, {total / 1e9:.2f} GB")

    counts = {"downloaded": 0, "cached": 0}
    for i, (entry, dest) in enumerate(wanted, 1):
        counts[download(entry, dest)] += 1
        if i % 20 == 0 or i == len(wanted):
            print(f"  {i}/{len(wanted)}  {counts['downloaded']} downloaded, "
                  f"{counts['cached']} already present")

    for name, expected in EXPECTED.items():
        path = args.dest / name
        if path.exists() and sha256(path) != expected:
            print(f"  note: {name} differs from the copy this release was verified against")

    print(f"\ndone. Set GPSVDI_DATA_ROOT={args.dest.resolve()}")


if __name__ == "__main__":
    main()
