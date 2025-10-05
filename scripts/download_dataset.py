import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_path(repo_id, remote_path, local_root, max_workers, use_symlinks):
    # Normalize folder-like paths to pattern "*"
    if remote_path == "" or remote_path.endswith("/"):
        allow = [f"{remote_path}*"] if remote_path != "" else None
    else:
        # single file
        allow = [remote_path]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=local_root,
        local_dir_use_symlinks=use_symlinks,
        resume_download=True,
        max_workers=max_workers,  # reduce concurrency: fewer requests per minute
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file or folder from a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        default="Blinorot/lensless_mic_librispeech",
        type=str,
        help="Repo ID on Hugging Face.",
    )
    parser.add_argument(
        "--remote-path",
        default="",
        type=str,
        help="Path in the repo to download. End with '/' for folders. Use '' to download all.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        default=None,
        type=str,
        help="Root local directory to mirror structure under.",
    )
    parser.add_argument(
        "--max-workers",
        default=1,
        help="Max number of downloading workers (Default: 1)",
    )
    parser.add_argument(
        "--use-symlinks",
        default=True,
        type=bool,
        help="Whether to use symlinks instead of copy (Default: True)",
    )

    args = parser.parse_args()
    download_path(
        args.repo_id,
        args.remote_path,
        args.local_dir,
        args.max_workers,
        args.use_symlinks,
    )
    print("Download complete.")
