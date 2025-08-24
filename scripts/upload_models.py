import argparse
import sys
from pathlib import Path

from huggingface_hub import HfFolder, create_repo, upload_folder


def main(args):
    repo_id = f"{args.username}/{args.repo_name}"

    token = HfFolder.get_token()
    if token is None:
        print(
            "Hugging Face token not found. Run `huggingface-cli login` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    local_dir = Path(args.local_dir).resolve()
    if not local_dir.is_dir():
        print(f"Local directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    # Create (or reuse) the repo
    print(f"Creating/reusing repo '{repo_id}'...")
    create_repo(
        repo_id=repo_id,
        token=token,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )

    # Optional: quick sanity check so you see which subfolders will be included
    subdirs = [p for p in local_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"No subdirectories found inside: {local_dir}", file=sys.stderr)
        sys.exit(1)

    # Inform the user what will be uploaded
    print("Scanning subfolders and checking for required files...")
    missing = []
    for sd in sorted(subdirs):
        cfg = sd / "config.yaml"
        ckpt = sd / args.checkpoint_filename
        if not cfg.exists() or not ckpt.exists():
            missing.append((sd.name, cfg.exists(), ckpt.exists()))
    if missing:
        print(
            "Warning: Some subfolders are missing required files (will be skipped by the allow patterns):"
        )
        for name, has_cfg, has_ckpt in missing:
            print(
                f"  - {name}: config.yaml={'OK' if has_cfg else 'MISSING'}, {args.checkpoint_filename}={'OK' if has_ckpt else 'MISSING'}"
            )

    # Upload ONLY the targeted files while preserving subfolder structure
    # Path patterns are evaluated relative to `folder_path`.
    allow_patterns = [
        "**/config.yaml",
        f"**/{args.checkpoint_filename}",
    ]

    print(f"\nUploading from '{local_dir}' to '{repo_id}' ...")
    print(f"  Allow patterns: {allow_patterns}")
    upload_folder(
        repo_id=repo_id,
        token=token,
        folder_path=str(local_dir),
        path_in_repo=".",  # keep subfolders at repo root
        allow_patterns=allow_patterns,
        commit_message=f"Upload selected files from {local_dir.name} subfolders",
    )

    print(f"\nDone. Repo: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload only config.yaml and a specific checkpoint file from each subdirectory to a Hugging Face repo."
    )
    parser.add_argument(
        "--username",
        default="Blinorot",
        help="Your Hugging Face username or organization.",
    )
    parser.add_argument(
        "--repo-name",
        default="lensless_mic_models",
        help="Target Hugging Face repo name.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Path to MainDir containing the SmallDir_* subfolders.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        default="checkpoint-epoch100.pth",
        help="Checkpoint filename to include from each subfolder (default: checkpoint-epoch100.pth).",
    )
    args = parser.parse_args()
    main(args)
