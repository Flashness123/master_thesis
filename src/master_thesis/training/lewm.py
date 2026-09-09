import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F

from omegaconf import DictConfig, OmegaConf, open_dict
from stable_pretraining import data as dt
from stable_worldmodel.wm.utils import save_pretrained

from master_thesis.models.lewm import SIGReg


class DiscreteActionOneHot:
    """
    Convert temporally grouped discrete action IDs to one-hot vectors.

    Example with frameskip=4 and six actions:

        [0, 2, 2, 3]

    becomes:

        4 x 6 one-hot values

    and is flattened to a 24-dimensional action vector.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert grouped discrete action IDs to one-hot vectors.

        SWM uses an undefined action at episode boundaries. For floating-point
        action spaces this is represented as NaN. For integer discrete action
        spaces, the NaN can become a negative integer sentinel during collection.

        Both cases are therefore treated as "no action" and encoded as an
        all-zero vector.
        """

        valid = (
            torch.isfinite(actions)
            & (actions >= 0)
            & (actions < self.num_actions)
        )

        safe_actions = torch.where(
            valid,
            actions,
            torch.zeros_like(actions),
        ).long()

        one_hot = F.one_hot(
            safe_actions,
            num_classes=self.num_actions,
        ).float()

        # Invalid/boundary actions become the zero vector rather than action 0.
        one_hot = one_hot * valid.unsqueeze(-1)

        return one_hot.flatten(start_dim=-2)


def build_image_transform(img_size: int):
    """
    LeWM image preprocessing:
    uint8 RGB -> float -> ImageNet normalization -> square resize.
    """
    image_stats = dt.dataset_stats.ImageNet

    to_image = dt.transforms.ToImage(
        **image_stats,
        source="pixels",
        target="pixels",
    )

    resize = dt.transforms.Resize(
        img_size,
        source="pixels",
        target="pixels",
    )

    return dt.transforms.Compose(
        to_image,
        resize,
    )


def build_dataset(cfg: DictConfig):
    dataset_name = cfg.dataset.path

    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        frameskip=cfg.dataset.frameskip,
        num_steps=(
            cfg.wm.history_size
            + cfg.wm.num_preds
        ),
        keys_to_load=list(
            cfg.dataset.keys_to_load
        ),
    )

    image_transform = build_image_transform(
        cfg.img_size,
    )

    action_transform = dt.transforms.WrapTorchTransform(
        DiscreteActionOneHot(
            cfg.dataset.action.num_actions,
        ),
        source="action",
        target="action",
    )

    dataset.transform = dt.transforms.Compose(
        image_transform,
        action_transform,
    )

    action_input_dim = (
        cfg.dataset.frameskip
        * cfg.dataset.action.num_actions
    )

    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            action_input_dim
        )

    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(dataset)}")
    print(
        f"Action encoder input: "
        f"{action_input_dim}"
    )

    return dataset


def lewm_forward(
    self,
    batch,
    stage,
    cfg,
):
    """
    LeWM training objective.

    Encode observations, predict the next latent state,
    and optimize prediction MSE + SIGReg.
    """
    history_size = cfg.wm.history_size
    num_preds = cfg.wm.num_preds

    output = self.model.encode(batch)

    embeddings = output["emb"]
    action_embeddings = output["act_emb"]

    context_embeddings = embeddings[
        :,
        :history_size,
    ]

    context_actions = action_embeddings[
        :,
        :history_size,
    ]

    targets = embeddings[
        :,
        num_preds:,
    ]

    predictions = self.model.predict(
        context_embeddings,
        context_actions,
    )

    prediction_loss = (
        predictions - targets
    ).pow(2).mean()

    sigreg_loss = self.sigreg(
        embeddings.transpose(0, 1)
    )

    loss = (
        prediction_loss
        + cfg.loss.sigreg.weight * sigreg_loss
    )

    self.log_dict(
        {
            f"{stage}/loss": loss.detach(),
            f"{stage}/pred_loss": prediction_loss.detach(),
            f"{stage}/sigreg_loss": sigreg_loss.detach(),
        },
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )

    return {
        "loss": loss,
        "pred_loss": prediction_loss,
        "sigreg_loss": sigreg_loss,
    }


def build_dataloaders(
    dataset,
    cfg: DictConfig,
):
    generator = torch.Generator().manual_seed(
        cfg.seed
    )

    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[
            cfg.train_split,
            1 - cfg.train_split,
        ],
        generator=generator,
    )

    loader_cfg = OmegaConf.to_container(
        cfg.loader,
        resolve=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        **loader_cfg,
        generator=generator,
    )

    val_loader_cfg = dict(loader_cfg)

    val_loader_cfg["shuffle"] = False
    val_loader_cfg["drop_last"] = False

    val_loader = torch.utils.data.DataLoader(
        val_set,
        **val_loader_cfg,
    )

    return train_loader, val_loader


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="experiment/pong_lewm",
)

def main(cfg: DictConfig) -> None:
    pl.seed_everything(
        cfg.seed,
        workers=True,
    )

    dataset = build_dataset(cfg)

    train_loader, val_loader = build_dataloaders(
        dataset,
        cfg,
    )

    model = hydra.utils.instantiate(
        cfg.model,
    )

    optimizers = {
        "model_optimizer": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
            },
            "interval": "epoch",
        }
    }

    module = spt.Module(
        model=model,
        sigreg=SIGReg(
            **cfg.loss.sigreg.kwargs,
        ),
        forward=partial(
            lewm_forward,
            cfg=cfg,
        ),
        optim=optimizers,
    )

    data_module = spt.data.DataModule(
        train=train_loader,
        val=val_loader,
    )

    stablewm_home = Path(
        os.environ["STABLEWM_HOME"]
    )

    run_dir = (
        stablewm_home
        / "runs"
        / cfg.run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    OmegaConf.save(
        cfg,
        run_dir / "config.yaml",
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=run_dir,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
    )

    manager()

    save_pretrained(
        model,
        run_name=cfg.run_name,
        config=cfg.model,
        filename="weights.pt",
    )


if __name__ == "__main__":
    main()