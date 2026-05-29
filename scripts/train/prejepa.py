import os
import time
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoVideoProcessor


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_img_preprocessor(source, target, img_size=224):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        spt.data.transforms.Resize(img_size, source=source, target=target),
    )


class VideoPipeline(spt.data.transforms.Transform):
    def __init__(self, processor, source='image', target='image'):
        super().__init__()
        self.processor, self.source, self.target = processor, source, target

    def __call__(self, x):
        frames = self.nested_get(x, self.source)
        self.nested_set(
            x,
            self.processor(frames, return_tensors='pt')[
                'pixel_values_videos'
            ].squeeze(0),
            self.target,
        )
        return x


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0:
            self._save(pl_module.model, epoch)
        if epoch == trainer.max_epochs:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


class ThroughputCallback(Callback):
    """Log training throughput as ``perf/samples_per_sec``."""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self._t0 = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._t0 is None:
            return
        dt = time.perf_counter() - self._t0
        if dt <= 0:
            return
        samples = self.batch_size * trainer.world_size
        pl_module.log(
            'perf/samples_per_sec', samples / dt, on_step=True, on_epoch=False
        )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _strip_action_dims(tensor, action_range):
    """Remove the action dimensions from the last axis."""
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def dinowm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()

    batch = self.model.encode(
        batch,
        target='emb',
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    embedding = batch['emb'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch['emb'][:, cfg.wm.num_preds :, ...].detach()

    # Per-modality losses
    pixels_dim = batch['pixels_emb'].size(-1)
    batch['pixels_loss'] = F.mse_loss(
        pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim]
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    # Actionless embeddings (for probes and total loss)
    batch['actionless_emb'] = _strip_action_dims(batch['emb'], action_range)
    batch['actionless_prev_emb'] = _strip_action_dims(embedding, action_range)
    batch['actionless_pred_emb'] = _strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_emb'] = _strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = F.mse_loss(
        batch['actionless_pred_emb'],
        batch['actionless_target_emb'].detach(),
    )

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    # 'fit' -> 'train', 'validate' -> 'val' for clean wandb panels
    prefix = {'fit': 'train', 'validate': 'val'}.get(stage, stage)
    is_train = stage == 'fit'

    # Headline metric: total latent-prediction MSE (the optimization target).
    # train -> per-step; val -> epoch-aggregated (the proxy we watch vs MPC).
    self.log(
        f'{prefix}/mse',
        batch['loss'].detach(),
        on_step=is_train,
        on_epoch=not is_train,
        prog_bar=True,
        sync_dist=True,
    )
    # Per-modality MSE components (pixels_loss, proprio_loss, ...).
    self.log_dict(
        {f'{prefix}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=is_train,
        on_epoch=not is_train,
        sync_dist=True,
    )
    return batch


class DinoWMModule(spt.Module):
    """``spt.Module`` that logs the true pre-clip gradient norm as ``train/grad_norm``.

    ``after_manual_backward`` fires right after backward and *before* spt clips
    gradients inside ``training_step``, so it sees the real gradient. Under AMP
    the grads are still loss-scaled here, so we divide by the ``GradScaler``
    scale to recover the true norm — without this, spt clips the still-scaled
    grads and the post-clip value is pinned at ``1/scale`` (meaningless).
    ``clip_grad_norm_`` with ``max_norm=inf`` only measures; it does not alter
    the grads, so spt's subsequent real clip is unaffected.
    """

    def after_manual_backward(self):
        scaler = getattr(self.trainer.precision_plugin, 'scaler', None)
        scale = scaler.get_scale() if scaler is not None else 1.0
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=float('inf')
        )
        self.log('train/grad_norm', norm / scale, on_step=True, on_epoch=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='./config', config_name='prejepa')
def run(cfg):
    # --- Dataset ---
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    keys_to_load = ['pixels'] + encoding_keys

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{cfg.dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get('encoding', {})
    ]

    if cfg.backbone.get('is_video_encoder', False):
        processor = AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source='pixels', target='pixels'),
            spt.data.transforms.Resize(
                cfg.image_size, source='pixels', target='pixels'
            ),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.wm.get('encoding', {}):
            if key not in dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = (
                dim if key != 'action' else dim * cfg.frameskip
            )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    encoder.eval()
    encoder.requires_grad_(False)

    is_cnn = hasattr(encoder.config, 'hidden_sizes')
    embed_dim = (
        encoder.config.hidden_sizes[-1]
        if is_cnn
        else encoder.config.hidden_size
    )
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    embed_dim += sum(cfg.wm.get('encoding', {}).values())

    if cfg.backbone.get('is_video_encoder', False):
        num_patches += num_patches * (cfg.n_steps // 4)

    with open_dict(cfg):
        cfg.model.predictor.dim = embed_dim
        cfg.model.predictor.num_patches = num_patches
        cfg.model.extra_encoders = {
            '_target_': 'torch.nn.ModuleDict',
            'modules': {
                key: {
                    '_target_': 'stable_worldmodel.wm.prejepa.module.Embedder',
                    'in_chans': cfg.extra_dims[key],
                    'emb_dim': int(cfg.wm.encoding[key]),
                }
                for key in cfg.wm.get('encoding', {})
            },
        }

    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder)

    world_model = DinoWMModule(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            'model_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer)}
        },
    )

    # --- Training ---
    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Run ID: {run_id}')

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    model_name = cfg.output_model_name
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
        # Tie the checkpoint dir to the wandb run ID (our model-ID convention):
        # `version` is the resolved run id (= `subdir` if set, else wandb's auto id).
        if logger.version:
            model_name = f'{cfg.output_model_name}_{logger.version}'
    logging.info(f'Checkpoint model name: {model_name}')

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=model_name,
                cfg=cfg.model,
                epoch_interval=5,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
            ThroughputCallback(batch_size=cfg.batch_size),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
