"""Phase 3 multimodal fusion scaffold.

Combines tabular (phase 1) predictions with image (phase 2) predictions.
"""

from dataclasses import dataclass


@dataclass
class Phase3Config:
    tabular_weight: float = 0.6
    image_weight: float = 0.4


def fuse_probabilities(tabular_prob: float, image_prob: float, cfg: Phase3Config) -> float:
    return (tabular_prob * cfg.tabular_weight) + (image_prob * cfg.image_weight)


def main() -> None:
    cfg = Phase3Config()
    print("Phase 3 scaffold ready")
    print(cfg)


if __name__ == "__main__":
    main()
