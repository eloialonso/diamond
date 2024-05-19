import hydra
from omegaconf import DictConfig, OmegaConf

from trainer import Trainer
from utils import skip_if_run_is_over


OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg: DictConfig):
    run(cfg)


@skip_if_run_is_over
def run(cfg):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
