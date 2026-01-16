from core.trainer_fm import Trainer
from core.inferencer_fm import Inferencer
from utils.tools import load_config

def main():
    cfg = load_config("configs/fm/config.yaml")

    if cfg.mode == 'train':
        trainer = Trainer(cfg)
        trainer.train()
    else:
        inferencer = Inferencer(cfg)
        inferencer.inference()

if __name__ == '__main__':
    main()