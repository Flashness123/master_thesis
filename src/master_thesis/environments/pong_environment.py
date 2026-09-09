from pathlib import Path

import hydra
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf
from stable_worldmodel.data.format import get_format


def create_world(cfg: DictConfig) -> swm.World:
    """
    Create the Pong world from configuration.

    The environment itself uses frameskip=1. Temporal subsampling for
    world-model training is handled later by the SWM Dataset.
    """
    world_config = OmegaConf.to_container(
        cfg.world,
        resolve=True,
    )

    return swm.World(**world_config)


def collect_dataset(cfg: DictConfig) -> None:
    """
    Collect random Pong trajectories using stable-worldmodel.
    """
    dataset_path = Path(cfg.collection.path)
    dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    world = create_world(cfg)

    try:
        policy = swm.policy.RandomPolicy(
            seed=cfg.policy.seed,
        )

        world.set_policy(policy)

        writer = get_format(
            cfg.collection.format
        ).open_writer(
            dataset_path,
            mode=cfg.collection.mode,
        )

        print(f"Environment: {cfg.world.env_name}")
        print(f"Episodes:    {cfg.collection.episodes}")
        print(f"Dataset:     {dataset_path}")

        world.collect(
            writer=writer,
            episodes=cfg.collection.episodes,
            seed=cfg.collection.seed,
        )

    finally:
        world.close()


@hydra.main(
    version_base=None,
    config_path="../../../configs/environment",
    config_name="pong",
)
def main(cfg: DictConfig) -> None:
    collect_dataset(cfg)


if __name__ == "__main__":
    main()