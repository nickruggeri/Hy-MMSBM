from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.data.data_io import DEFAULT_DATA_DIR


def download_file(url: str, output: Path) -> None:
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit="B", unit_scale=True)
    with open(output, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)


def download_and_unzip(
    url: str, out_zip: Path, out_dir: Path, descr: str = "data"
) -> None:
    if not out_zip.exists():
        print(f"{descr}: downloading to:\n", out_zip)
        download_file(url, out_zip)
    else:
        print("File already exists:\n", out_zip)

    print(f"Unzipping to:\n", out_dir)
    with zipfile.ZipFile(out_zip, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    print(f"Removing zipped file:\n", out_zip)
    out_zip.unlink()


if __name__ == "__main__":
    DEFAULT_DATA_DIR.mkdir(parents=False, exist_ok=True)

    real_data_url = "https://edmond.mpdl.mpg.de/api/access/datafile/199947"
    zip_real_data = DEFAULT_DATA_DIR / "preprocessed_real_data.zip"
    download_and_unzip(
        real_data_url,
        zip_real_data,
        DEFAULT_DATA_DIR,
        descr="Real data",
    )

    print("\n")

    synthetic_data_url = "https://edmond.mpdl.mpg.de/api/access/datafile/200133"
    zip_synthetic_data = DEFAULT_DATA_DIR / "synthetic_data.zip"
    download_and_unzip(
        synthetic_data_url,
        zip_synthetic_data,
        DEFAULT_DATA_DIR,
        descr="Synthetic data",
    )
