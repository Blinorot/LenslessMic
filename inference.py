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


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    codec_dataset = instantiate(config.codec)  # cpu
    codec_inferencer = instantiate(config.codec).to(device)  # on device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, codec_dataset)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    # get metrics
    if config.metrics is not None:
        metrics = instantiate(config.metrics)
    else:
        metrics = None

    # save_path for model predictions
    if config.inferencer.save_path is None:
        save_path = ROOT_PATH / "data" / "datasets" / "reconstructed"
    else:
        save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
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

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("resolve_class", resolve_class)
    main()
