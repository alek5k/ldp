"""Upload/download the exact main-paper selection in paper_upload_files.json.

Zarr directories are archived with images and metadata intact. Nothing is deleted
from the Hub or the source directories. Importing this module never uploads data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path


def repository_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "paper_upload_files.json").is_file():
            return parent
    raise FileNotFoundError("Cannot find paper_upload_files.json beside the repository root")


def checked_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative path: {value}")
    return path


def source_files(source):
    if source.is_file():
        return [source]
    if source.is_dir() and source.suffix == ".zarr":
        files = sorted(p for p in source.rglob("*") if p.is_file())
        if not (source / ".zgroup").is_file():
            raise ValueError(f"Not a Zarr group: {source}")
        return files
    raise FileNotFoundError(f"Missing selected input: {source}")


def fingerprint(files, source):
    return [[p.relative_to(source).as_posix(), p.stat().st_size, p.stat().st_mtime_ns]
            for p in files]


def stage_file(source, destination, cache):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        if destination.exists() and os.path.samefile(source, destination):
            return
        temporary = destination.with_name(destination.name + ".partial")
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
        return
    files = source_files(source)
    signature = fingerprint(files, source)
    marker = cache / (hashlib.sha256(str(source.resolve()).encode()).hexdigest() + ".json")
    if destination.is_file() and marker.is_file():
        previous = json.loads(marker.read_text())
        if previous.get("source") == str(source.resolve()) and previous.get("files") == signature:
            return
    # Compression already happens in the Zarr chunks; store them without an
    # additional lossy conversion or a slow redundant compression pass.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".partial", delete=False) as f:
        temporary = Path(f.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in files:
                archive.write(path, arcname=path.relative_to(source.parent).as_posix())
        if fingerprint(source_files(source), source) != signature:
            raise ValueError(f"Dataset changed while archiving: {source}")
        temporary.replace(destination)
        marker.write_text(json.dumps({"source": str(source.resolve()), "files": signature}))
    finally:
        temporary.unlink(missing_ok=True)


def download(api, selection, repo_id, root):
    from huggingface_hub import hf_hub_download
    for item in selection:
        destination = root / checked_relative(item["source"])
        print(f"Downloading {item['remote']}", flush=True)
        cached = Path(hf_hub_download(repo_id, item["remote"], repo_type="dataset"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix == ".zarr":
            with zipfile.ZipFile(cached) as archive:
                for name in archive.namelist():
                    relative = checked_relative(name)
                    if relative.parts[0] != destination.name:
                        raise ValueError(f"Unexpected archive member: {name}")
                archive.extractall(destination.parent)
        else:
            shutil.copy2(cached, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=repository_root())
    parser.add_argument("--fallback-root", type=Path, action="append", default=[],
                        help="also look under this repository root for archived source files")
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--category", choices=["all", "demonstration", "checkpoint", "inference", "config"], default="all")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    root = args.source_root.expanduser().absolute()
    manifest = json.loads((repository_root() / "paper_upload_files.json").read_text())
    selection = [r for r in manifest["files"] if args.category == "all" or r["category"] == args.category]
    if args.category == "checkpoint":
        selection += [r for r in manifest["files"] if r["category"] == "config"]
    remotes = [r["remote"] for r in selection]
    if len(set(remotes)) != len(remotes):
        raise ValueError("Duplicate upload destinations")
    for item in selection:
        checked_relative(item["source"])
        checked_relative(item["remote"])
    if args.download:
        if args.dry_run:
            print(json.dumps(selection, indent=2))
            return
        download(None, selection, manifest["repo_id"], root)
        return
    total = 0
    sources = {}
    for item in selection:
        candidates = [base.expanduser() / item["source"] for base in [root, *args.fallback_root]]
        source = next((p for p in candidates if p.exists()), candidates[0])
        sources[item["source"]] = source
        files = source_files(source)
        total += sum(p.stat().st_size for p in files)
    print(f"{root.name}: {dict(Counter(r['category'] for r in selection))}; "
          f"{total / 2**30:.3f} GiB; destination {manifest['repo_id']}", flush=True)
    if args.dry_run:
        print("Dry run: no staging or upload", flush=True)
        return
    stage = (args.stage_dir or Path(tempfile.gettempdir()) / "tcdp-paper-hf-upload" / root.name).absolute()
    if stage == root or root in stage.parents:
        raise ValueError("Stage directory must be outside the source repository")
    cache = stage / ".cache" / "tcdp-archives"
    cache.mkdir(parents=True, exist_ok=True)
    for item in selection:
        print(f"Staging {item['remote']}", flush=True)
        stage_file(sources[item["source"]], stage / item["remote"], cache)
    card = repository_root() / "PAPER_DATASET_CARD.md"
    allow = list(remotes)
    if card.is_file():
        shutil.copy2(card, stage / "README.md")
        allow.append("README.md")
    size = sum((stage / path).stat().st_size for path in allow)
    print(f"Prepared {len(allow)} files, {size / 2**30:.3f} GiB in {stage}", flush=True)
    if args.prepare_only:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    # Repositories must already exist. Preserve their visibility and contents.
    info = api.repo_info(manifest["repo_id"], repo_type="dataset")
    print(f"Uploading to existing {'private' if info.private else 'public'} dataset {manifest['repo_id']}", flush=True)
    api.upload_large_folder(repo_id=manifest["repo_id"], repo_type="dataset",
                            folder_path=str(stage), allow_patterns=allow,
                            private=info.private, num_workers=args.num_workers,
                            print_report_every=60)
    print("Upload complete", flush=True)


if __name__ == "__main__":
    main()
