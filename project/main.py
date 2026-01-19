from pathlib import Path

import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy

from project.config import get_config
from project.dataset import VTGDataModule
from project.model import MyModule
from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":
    cfg = get_config()
    strategy = "ddp" if cfg.world_size > 1 else "auto"
    if cfg.deepspeed:
        strategy = DeepSpeedStrategy(config=cfg.deepspeed_config)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=cfg.checkpoint_every_n_val,
        enable_checkpointing=True,
        accelerator="gpu",
        strategy=strategy,
        devices="auto",
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        precision=cfg.precision,
        val_check_interval=cfg.val_check_interval,
        num_sanity_val_steps=cfg.num_sanity_steps,
        
    )

    with trainer.init_module():
        module = MyModule(cfg)

    datamodule = VTGDataModule(cfg, mode=cfg.mode, tokenizer=module.tokenizer)
    module.num_frames = datamodule.num_frames

    ckpt_dir = Path(cfg.checkpoint_dir)

    if cfg.mode == "fit":
        trainer.fit(model=module, datamodule=datamodule)
    elif cfg.mode == "validate":
        trainer.validate(model=module, datamodule=datamodule)
    else:
        trainer.predict(model=module, datamodule=datamodule)
