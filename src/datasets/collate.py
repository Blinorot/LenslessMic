from collections import defaultdict

import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = defaultdict(list)

    for elem in dataset_items:
        # audio is already cropped/padded, no need to do it here
        result_batch["audio"].append(elem["audio"].unsqueeze(0))
        result_batch["lensed_codec_video"].append(
            elem["lensed_codec_video"].unsqueeze(0)
        )
        result_batch["lensless_codec_video"].append(
            elem["lensless_codec_video"].unsqueeze(0)
        )
        result_batch["lensless_psf"].append(elem["lensless_psf"].unsqueeze(0))
        result_batch["min_vals"].append(elem["min_vals"])  # already with B
        result_batch["max_vals"].append(elem["max_vals"])  # already with B
        result_batch["audio_path"].append(elem["audio_path"])
        result_batch["text"].append(elem["text"])

    result_batch["audio"] = torch.cat(result_batch["audio"], dim=0)
    result_batch["lensed_codec_video"] = torch.cat(
        result_batch["lensed_codec_video"], dim=0
    )
    result_batch["lensless_codec_video"] = torch.cat(
        result_batch["lensless_codec_video"], dim=0
    )
    result_batch["lensless_psf"] = torch.cat(result_batch["lensless_psf"], dim=0)
    result_batch["min_vals"] = torch.cat(result_batch["min_vals"], dim=0)
    result_batch["max_vals"] = torch.cat(result_batch["max_vals"], dim=0)

    return result_batch
