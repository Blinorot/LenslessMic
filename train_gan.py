import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import TrainerGAN
from src.utils.init_utils import (
    resolve_class,
    set_random_seed,
    setup_saving_and_logging,
)

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baselinegan")
def main(config):
    """
    Main script for training with GAN-losses. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    codec_dataset = instantiate(config.codec)  # cpu
    codec_trainer = instantiate(config.codec).to(device)  # on device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device, codec_dataset)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    discriminator = instantiate(config.discriminator).to(device)
    logger.info(model)
    logger.info(discriminator)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_G = instantiate(config.optimizer_G, params=trainable_params)
    optimizer_D = instantiate(config.optimizer_D, params=discriminator.parameters())
    lr_scheduler_G = instantiate(config.lr_scheduler_G, optimizer=optimizer_G)
    lr_scheduler_D = instantiate(config.lr_scheduler_D, optimizer=optimizer_D)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = TrainerGAN(
        model=model,
        discriminator=discriminator,
        codec=codec_trainer,
        criterion=loss_function,
        metrics=metrics,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        lr_scheduler_G=lr_scheduler_G,
        lr_scheduler_D=lr_scheduler_D,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    OmegaConf.register_new_resolver("resolve_class", resolve_class)
    main()
