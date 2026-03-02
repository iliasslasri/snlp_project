import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from src.utils import set_seed, setup_logging
from src.robust_quantization import train_quantizer

@hydra.main(version_base=None, config_path="configs", config_name="quantization")
def main(cfg: DictConfig):
    if cfg.get('seed'):
        set_seed(cfg.seed)
    
    out_dir = cfg.training.checkpoint_dir if 'training' in cfg else 'outputs'
    setup_logging(out_dir)
    
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logging.info("Starting Quantization Training...")
    
    train_quantizer(cfg)

if __name__ == "__main__":
    main()
