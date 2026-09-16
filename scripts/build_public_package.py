"""Build the curated public OncoLncAI source package."""

from __future__ import annotations

import argparse
from pathlib import Path

from oncolncai.deployment import build_public_package, verify_public_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--destination", type=Path, default=Path("dist/oncolncai_public"))
    parser.add_argument("--manifest", type=Path, default=Path("deployment/public_files.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if args.verify and destination.is_dir():
        inventory = verify_public_package(destination)
    else:
        manifest = args.manifest if args.manifest.is_absolute() else source / args.manifest
        inventory = build_public_package(source, destination, manifest)
    print(f"public package: {destination}")
    print(f"source files: {len(inventory.files)}")
    print(f"aggregate sha256: {inventory.aggregate_sha256}")


if __name__ == "__main__":
    main()
