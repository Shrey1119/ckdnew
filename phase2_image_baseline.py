"""Phase 2 image baseline scaffold.

This module is intentionally lightweight so phase 1 can stay fast and accurate.
Use this as the next step for image-only kidney class prediction.
"""

from dataclasses import dataclass


@dataclass
class Phase2Config:
    model_name: str = "resnet18"
    image_size: int = 224
    epochs: int = 10
    batch_size: int = 16


def main() -> None:
    config = Phase2Config()
    print("Phase 2 scaffold ready")
    print(config)


if __name__ == "__main__":
    main()
