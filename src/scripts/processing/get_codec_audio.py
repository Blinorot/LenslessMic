import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.init_utils import resolve_class
from src.utils.io_utils import ROOT_PATH


@hydra.main(
    version_base=None,
    config_path=str(ROOT_PATH / "src/configs/scripts"),
    config_name="convert_dataset",
)
def convert_dataset(config):
    dataset = instantiate(config.dataset)
    codec = instantiate(config.codec)

    dataset.save_in_audio_format(codec=codec)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("resolve_class", resolve_class)
    convert_dataset()
