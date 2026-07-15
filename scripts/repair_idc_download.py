"""Repair IDC downloads left as .dcm.tmp without deleting any files."""

from __future__ import annotations

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pydicom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", required=True, type=Path)
    parser.add_argument("--s5cmd", required=True, type=Path)
    parser.add_argument(
        "--endpoint-url",
        default="https://s3.amazonaws.com",
        choices=("https://s3.amazonaws.com", "https://storage.googleapis.com"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-count", type=int, default=10)
    return parser.parse_args()


def validate_dicom(path: Path) -> None:
    dataset = pydicom.dcmread(path)
    _ = dataset.pixel_array


def main() -> int:
    args = parse_args()
    collection_root = args.download_root / "cbis_ddsm"
    metadata_path = args.download_root / "metadata" / "metadata.csv"
    if not collection_root.is_dir():
        raise FileNotFoundError(f"Collection directory not found: {collection_root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"IDC metadata not found: {metadata_path}")
    if not args.s5cmd.is_file():
        raise FileNotFoundError(f"s5cmd executable not found: {args.s5cmd}")

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_series = {row["SeriesInstanceUID"]: row for row in rows}

    temporary_files = sorted(collection_root.rglob("*.dcm.tmp"))
    print(f"Found {len(temporary_files)} temporary DICOM files.")
    jobs: list[tuple[int, Path, Path, str]] = []
    for index, temporary_path in enumerate(temporary_files, start=1):
        final_path = Path(str(temporary_path)[: -len(".tmp")])
        if final_path.is_file():
            validate_dicom(final_path)
            print(f"[{index}/{len(temporary_files)}] already valid: {final_path}")
            continue

        header = pydicom.dcmread(temporary_path, stop_before_pixels=True)
        series_uid = str(header.SeriesInstanceUID)
        if series_uid not in by_series:
            raise RuntimeError(f"Series UID is absent from IDC metadata: {series_uid}")
        row = by_series[series_uid]
        object_name = temporary_path.name[: -len(".tmp")]
        source = row["OriginalS5cmdURI"].replace("*", object_name)
        jobs.append((index, temporary_path, final_path, source))

    def repair(job: tuple[int, Path, Path, str]) -> Path:
        index, _temporary_path, final_path, source = job
        print(f"[{index}/{len(temporary_files)}] downloading {source}", flush=True)
        subprocess.run(
            [
                str(args.s5cmd),
                "--no-sign-request",
                "--endpoint-url",
                args.endpoint_url,
                "--retry-count",
                str(args.retry_count),
                "cp",
                source,
                str(final_path),
            ],
            check=True,
        )
        validate_dicom(final_path)
        print(f"[{index}/{len(temporary_files)}] validated {final_path}", flush=True)
        return final_path

    repaired = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(repair, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            repaired += 1

    print(f"Validated {len(temporary_files)} final DICOM files; repaired {repaired}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
