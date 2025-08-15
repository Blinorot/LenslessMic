from pathlib import Path

import wget
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

ROOT_DIR = Path(__file__).absolute().resolve().parent.parent

ORIGINAL_URL = "https://huggingface.co/ZhenYe234/xcodec/resolve/main/xcodec_hubert_general_audio.pth"


def main():
    local_root = (
        ROOT_DIR / "data" / "xcodec_exps" / "32x32_80_16khz_xcodec_hubert_general_audio"
    )
    local_root.mkdir(exist_ok=True, parents=True)

    print("Downloading original X-codec weights")
    wget.download(ORIGINAL_URL, str(local_root / "weights.pth"))

    print("Download complete.")


if __name__ == "__main__":
    main()
