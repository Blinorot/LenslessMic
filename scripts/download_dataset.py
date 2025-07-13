import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download


def download_single_file(repo_id, remote_path, local_root):
    local_path = os.path.join(local_root, remote_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"Downloading file '{remote_path}'...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id, filename=remote_path, repo_type="dataset"
    )

    shutil.copy2(downloaded_path, local_path)
    print(f"Saved to: {local_path}")


def download_folder(repo_id, remote_folder, local_root):
    print(f"Downloading folder '{remote_folder}' from '{repo_id}'...")

    # Download only files matching the remote folder
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{remote_folder}*"] if remote_folder != "" else None,
    )

    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
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
    local_root = Path(args.local_dir).resolve()
    local_root.mkdir(exist_ok=True, parents=True)

    if remote_path.endswith("/") or remote_path == "":
        download_folder(repo_id, remote_path, str(local_root))
    else:
        download_single_file(repo_id, remote_path, str(local_root))

    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file or folder from a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        default="Blinorot/lensless_mic_librispeech",
        help="Repo ID on Hugging Face.",
    )
    parser.add_argument(
        "--remote-path",
        default="",
        help="Path in the repo to download. End with '/' for folders. Use '' to download all.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        default=None,
        help="Root local directory to mirror structure under.",
    )

    args = parser.parse_args()
    main(args)
