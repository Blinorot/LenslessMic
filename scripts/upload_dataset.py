import argparse
import os

from huggingface_hub import HfFolder, create_repo, upload_folder


def main(args):
    repo_id = f"{args.username}/{args.repo_name}"

    token = HfFolder.get_token()
    if token is None:
        print(
            "Hugging Face token not found. Run `huggingface-cli login` or set the token programmatically."
        )
        return

    print(f"Creating dataset repo '{repo_id}'...")
    create_repo(
        repo_id=repo_id,
        token=token,
        private=args.private,
        repo_type="dataset",
        exist_ok=True,
    )

    allow_patterns = None
    if args.ignore_unused_audio:
        allow_patterns = []

        ref_dir = args.ignore_reference_dir
        assert ref_dir is not None

        suffix = ".mkv"

        for file in os.listdir(ref_dir):
            if file.endswith(suffix):
                object_name = file[: -len(suffix)]
                allow_patterns.append(f"{object_name}.flac")
                allow_patterns.append(f"{object_name}.wav")
                allow_patterns.append(f"{object_name}.mp3")
                # add transcription
                allow_patterns.append(f"{object_name}.txt")

    # Upload directory to desired path in the repo
    print(
        f"Uploading folder '{args.local_dir}' to '{repo_id}' under path '{args.remote_dir or './'}'..."
    )
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=args.local_dir,
        path_in_repo=args.remote_dir or ".",  # preserve directory name if needed
        token=token,
        allow_patterns=allow_patterns,
    )

    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a dataset (or part of one) to Hugging Face Hub."
    )
    parser.add_argument(
        "--username",
        default="Blinorot",
        help="Your Hugging Face username or organization.",
    )
    parser.add_argument(
        "--repo-name",
        default="lensless_mic_librispeech",
        help="Name of the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory with the dataset or subfolder to upload.",
    )
    parser.add_argument(
        "--remote-dir",
        default=None,
        help="Path in the repo to upload to (e.g., '16x16_130_16khz/lensed').",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repo if set."
    )
    parser.add_argument(
        "--ignore-unused-audio",
        action="store_true",
        help="Ignore audio without lensed/lensless copy.",
    )
    parser.add_argument(
        "--ignore-reference-dir",
        default=None,
        help="Which directory with .mkv files to use as reference to ignore audio without copy.",
    )

    args = parser.parse_args()
    main(args)
