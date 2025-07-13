import argparse
import os
import shutil
from pathlib import Path

import wget
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

ROOT_DIR = Path(__file__).absolute().resolve().parent.parent

ORIGINAL_URL = "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth"


def download_single_file(repo_id, remote_path, local_root):
    local_path = os.path.join(local_root, remote_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"Downloading file '{remote_path}'...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id, filename=remote_path, repo_type="model"
    )

    # Copy from cache to structured path
    shutil.copy2(downloaded_path, local_path)
    print(f"Saved to: {local_path}")


def download_folder(repo_id, remote_folder, local_root):
    print(f"Downloading folder '{remote_folder}' from '{repo_id}'...")

    # Snapshot to get matching files
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{remote_folder}*"] if remote_folder != "" else None,
    )

    # Get all files in the snapshot
    files = list_repo_files(repo_id=repo_id, repo_type="model")
    for f in files:
        if f.startswith(remote_folder):
            src_path = os.path.join(snapshot_path, f)
            dst_path = os.path.join(local_root, f)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            shutil.copy2(src_path, dst_path)
            print(f"Saved: {dst_path}")


def main(args):
    repo_id = args.repo_id
    remote_path = args.remote_path
    local_root = ROOT_DIR / "data" / "dac_exps"
    local_root.mkdir(exist_ok=True, parents=True)
    local_root = str(local_root)

    if remote_path.endswith("/") or remote_path == "":
        download_folder(repo_id, remote_path, local_root)
    else:
        download_single_file(repo_id, remote_path, local_root)

    if args.download_original:
        print("Downloading original DAC weights")
        local_dir = Path(local_root) / "32x32_120_16khz_original" / "latest" / "dac"
        local_dir.mkdir(exist_ok=True, parents=True)
        wget.download(ORIGINAL_URL, str(local_dir / "weights.pth"))

    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file or folder from Hugging Face and mirror its structure."
    )
    parser.add_argument(
        "--repo-id",
        default="Blinorot/dac_finetuned_librispeech",
        help="Repo ID on Hugging Face.",
    )
    parser.add_argument(
        "--remote-path",
        default="16x16_130_16khz/latest/dac/weights.pth",
        help="Path in the repo to download. End with '/' for folders.",
    )
    parser.add_argument(
        "--download_original", action="store_true", help="Download original dac weights"
    )
    args = parser.parse_args()

    main(args)
