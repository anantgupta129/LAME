import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPVisionModel

# class CLIPVisionTower(nn.Module):
#     def __init__(self, model_id: str = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M") -> None:
#         super().__init__()

#         self.model_id = model_id
#         self._load_model()

#     def _load_model(self) -> None:
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.model_id)
#         self.vision_tower = CLIPModel.from_pretrained(self.model_id)
#         self.vision_tower.requires_grad_(False)

#     @torch.inference_mode()
#     def forward(self, img0: Image) -> torch.Tensor:
#         img1 = self.image_processor(images=img0, return_tensors="pt")["pixel_values"].half().to(self.device)
#         outputs = self.vision_tower.get_image_features(pixel_values=img1)
#         return outputs

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device


class CLIPVisionTower(nn.Module):
    def __init__(
        self, mm_vision_select_layer: int, mm_vision_select_feature: str, model_id: str
    ) -> None:
        super().__init__()

        self.model_id = model_id
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature

        self._load_model()

    def _load_model(self) -> None:
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_id)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.model_id)
        self.vision_tower.requires_grad_(False)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.inference_mode()
    def forward(self, img0: torch.Tensor) -> torch.Tensor:
        image_forward_outs = self.vision_tower(
            img0.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(img0.dtype)
        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
