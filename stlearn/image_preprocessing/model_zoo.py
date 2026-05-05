from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class Model:
    _WEIGHTS: ClassVar[dict] = {
        ("resnet50", "v1"): models.ResNet50_Weights.IMAGENET1K_V1,
        ("resnet50", "v2"): models.ResNet50_Weights.IMAGENET1K_V2,
        ("vgg16", "v1"): models.VGG16_Weights.IMAGENET1K_V1,
        ("inception_v3", "v1"): models.Inception_V3_Weights.IMAGENET1K_V1,
        ("convnext_tiny", "v1"): models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        ("convnext_small", "v1"): models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        ("convnext_base", "v1"): models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        ("convnext_large", "v1"): models.ConvNeXt_Large_Weights.IMAGENET1K_V1,
    }

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

    def _resolve_weights(self):
        legal = {v for b, v in self._WEIGHTS if b == self.base}
        if not legal:
            raise ValueError(f"{self.base!r} is not a valid model")
        if self.weights_version not in legal:
            raise ValueError(
                f"weights={self.weights_version!r} is not available for "
                f"base={self.base!r}. Available versions: {sorted(legal)}",
            )
        return self._WEIGHTS[(self.base, self.weights_version)]

    def _load_model(self):
        if self.base == "resnet50":
            weights = self._resolve_weights()
            net = models.resnet50(weights=weights)
            net.fc = nn.Identity()  # → 2048-d pooled features
            weight_transforms = weights.transforms()

        elif self.base == "vgg16":
            weights = self._resolve_weights()
            base_net = models.vgg16(weights=weights)
            # Keras include_top=False, pooling="avg" = conv stack + global avg pool.
            net = nn.Sequential(
                base_net.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            weight_transforms = weights.transforms()

        elif self.base == "inception_v3":
            weights = self._resolve_weights()
            # Load the pretrained weights
            net = models.inception_v3(weights=weights, aux_logits=True)
            # Disable the aux branch at forward time - lowercase
            net.aux_logits = False
            net.fc = nn.Identity()
            weight_transforms = weights.transforms()

        elif self.base in (
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
        ):
            weights = self._resolve_weights()
            builder = getattr(models, self.base)
            net = builder(weights=weights)
            # C = 768/768/1024/1536 for Tiny/Small/Base/Large
            # ConvNeXt's head is: avgpool → LayerNorm2d → Flatten → Linear(C, 1000)
            # We want everything except the final Linear, so the model outputs
            # C-d pooled features (analogous to include_top=False, pooling="avg").
            net.classifier[2] = nn.Identity()
            weight_transforms = weights.transforms()
        else:
            raise ValueError(f"{self.base!r} is not a valid model")

        # Bridge NHWC numpy → CHW float tensor, then apply the weights'
        # own preprocessing (resize, normalize, etc.). We use Compose just
        # to chain ToTensor with the weights transform.
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                weight_transforms,
            ],
        )
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

        outputs = []
        for start in range(0, len(x), self.batch_size):
            chunk = x[start : start + self.batch_size]
            batch = torch.stack([self.preprocess(img) for img in chunk]).to(self.device)
            outputs.append(self.model(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)
