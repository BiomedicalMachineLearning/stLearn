import torch
import torch.nn as nn
from torchvision import models, transforms


class Model:
    __name__ = "CNN base model"

    def __init__(self, base, batch_size=32):
        self.base = base
        self.batch_size = batch_size
        self.device = torch.device(
            (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            ),
        )
        self.model, self.preprocess = self._load_model()
        self.model = self.model.to(self.device).eval()

    def _load_model(self):
        if self.base == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            net = models.resnet50(weights=weights)
            net.fc = nn.Identity()  # → 2048-d pooled features
            weight_transforms = weights.transforms()

        elif self.base == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1
            base_net = models.vgg16(weights=weights)
            # Keras include_top=False, pooling="avg" = conv stack + global avg pool.
            net = nn.Sequential(
                base_net.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            weight_transforms = weights.transforms()

        elif self.base == "inception_v3":
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            net = models.inception_v3(weights=weights, aux_logits=True)
            net.fc = nn.Identity()
            net.AuxLogits = None  # disable aux head
            weight_transforms = weights.transforms()

        elif self.base == "xception":
            try:
                import timm
                from timm.data import create_transform, resolve_data_config
            except ImportError as e:
                raise ImportError(
                    "xception backbone requires the optional `timm` dependency. "
                    "Install with: pip install timm",
                ) from e
            net = timm.create_model("xception", pretrained=True, num_classes=0)
            # timm's equivalent of weights.transforms(): read the model's
            # pretrained config and build the matching preprocessing pipeline.
            data_config = resolve_data_config({}, model=net)
            weight_transforms = create_transform(**data_config)

        else:
            raise ValueError(f"{self.base} is not a valid model")

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

        batch = torch.stack([self.preprocess(img) for img in x]).to(self.device)
        out = self.model(batch)
        return out.cpu().numpy()
