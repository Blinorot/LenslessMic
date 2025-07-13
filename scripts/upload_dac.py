import argparse

from huggingface_hub import HfFolder, create_repo, upload_folder


def main(args):
    repo_id = f"{args.username}/{args.repo_name}"

    # Log in (assumes you have HF token configured)
    token = HfFolder.get_token()
    if token is None:
        print("Hugging Face token not found. Run `huggingface-cli login` first.")
        return

    print(f"Creating repo '{repo_id}'...")
    create_repo(
        repo_id=repo_id,
        token=token,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )

    # Upload the folder
    print(f"Uploading folder '{args.local_dir}' to '{repo_id}'...")
    upload_folder(
        repo_id=repo_id,
        folder_path=args.local_dir,
        path_in_repo=".",  # uploads the folder contents to root
        token=token,
    )

    print(f"Upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a directory of model weights to Hugging Face Hub."
    )
    parser.add_argument(
        "--username",
        default="Blinorot",
        help="Your Hugging Face username or organization name.",
    )
    parser.add_argument(
        "--repo-name",
        default="dac_finetuned_librispeech",
        help="Name of the Hugging Face repo.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        default=None,
        help="Local directory with model files.",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repo if set."
    )
    args = parser.parse_args()
    main(args)
