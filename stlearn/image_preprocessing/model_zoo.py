from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

CNN_BASE = Literal[
    "resnet50",
    "vgg16",
    "inception_v3",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]
WEIGHTS_VERSION = Literal["v1", "v2"]


class Model:
    def __init__(self, base: str, batch_size: int = 32, weights: str = "v1"):
        self.base = base
        self.batch_size = batch_size
        self.weights_version = weights
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
        self.device = torch.device(device_str)
        self.model, self.preprocess = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        weights = models.get_model_weights(self.base)[
            f"IMAGENET1K_{self.weights_version.upper()}"
        ]
        net = models.get_model(self.base, weights=weights)

        # Drop each network's classification layer so it outputs pooled
        # features (as Keras include_top=False, pooling="avg").
        if self.base == "resnet50":
            # 2048-d
            net.fc = nn.Identity()
        elif self.base == "vgg16":
            net = nn.Sequential(
                net.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )  # 512-d
        elif self.base == "inception_v3":
            # The pretrained weights include the aux branch; skip it in forward.
            net.aux_logits = False
            # 2048-d
            net.fc = nn.Identity()
        elif self.base.startswith("convnext_"):
            # Head: avgpool -> LayerNorm2d -> Flatten -> Linear(C, 1000); drop the
            # Linear. C = 768/768/1024/1536 for tiny/small/base/large.
            net.classifier[2] = nn.Identity()
        else:
            raise ValueError(f"No feature head defined for {self.base!r}")

        # Bridge NHWC numpy -> CHW float tensor, then apply the weights' own
        # preprocessing (resize, crop, normalise).
        preprocess = transforms.Compose([transforms.ToTensor(), weights.transforms()])
        return net, preprocess

    @torch.inference_mode()
    def predict(self, x):
        if x.ndim != 4:
            raise ValueError(f"expected NHWC array, got shape {x.shape}")
        if x.shape[-1] != 3:
            raise ValueError(
                f"expected 3 channels (RGB) in last axis, got {x.shape[-1]}. "
                f"For RGBA images, drop the alpha channel; for grayscale, "
                f"replicate to 3 channels."
            )
        if x.dtype != np.uint8:
            raise TypeError(
                f"expected uint8 pixel values (0-255), got {x.dtype}; "
                "ToTensor only rescales uint8 input"
            )
        outputs = []
        for start in range(0, len(x), self.batch_size):
            chunk = x[start : start + self.batch_size]
            batch = torch.stack([self.preprocess(img) for img in chunk]).to(self.device)
            outputs.append(self.model(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)
