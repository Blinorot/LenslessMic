import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import resolve_class, set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    version_base=None, config_path="src/configs", config_name="calculate_metrics"
)
def main(config):
    """
    Main script for calculating metrics. Instantiates metrics and
    dataloaders. Runs Inferencer to calculate metrics.
    Predictions are pre-calculated by inference.py

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    device = "cpu"

    codec_dataset = instantiate(config.codec)  # cpu
    codec_inferencer = instantiate(config.codec).to(device)  # on device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, codec_dataset)

    # build model architecture, then print to console
    # model = instantiate(config.model).to(device)
    # print(model)

    # get metrics
    if config.metrics is not None:
        metrics = instantiate(config.metrics)
    else:
        metrics = None

    # save_path for model predictions
    if config.inferencer.save_path is None:
        save_path = ROOT_PATH / "data" / "datasets" / "reconstructed"
    else:
        save_path = Path(config.inferencer.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    # save_path for model predictions
    if config.inferencer.metrics_path is None:
        metrics_path = ROOT_PATH / "data" / "metrics" / "reconstructed"
    else:
        metrics_path = Path(config.inferencer.metrics_path)

    metrics_path = metrics_path / config.inferencer.dataset_tag

    metrics_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=None,
        codec=codec_inferencer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        dataset_tag=config.inferencer.dataset_tag,
        model_tag=config.inferencer.model_tag,
        metrics=metrics,
        skip_model_load=config.inferencer.skip_model_load,
    )

    logs = inferencer.run_inference(calculate_metrics=True)

    for part in logs.keys():
        part_metrics_path = metrics_path / part / config.inferencer.model_tag
        part_metrics_path.mkdir(exist_ok=True, parents=True)
        with open(part_metrics_path / "metrics.txt", mode="w") as f:
            keys = sorted(logs[part].keys())
            for key in keys:
                value = logs[part][key]
                full_key = part + "_" + key
                f.write(f"{full_key}: {value:.3f}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("resolve_class", resolve_class)
    main()
