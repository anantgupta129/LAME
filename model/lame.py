from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class ImageProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3),
            nn.GELU(),
            nn.GroupNorm(1, 256),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.GELU(),
            nn.GroupNorm(1, 256),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.GELU(),
            nn.GroupNorm(1, 512),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.GELU(),
            nn.GroupNorm(1, 512),
        )
        self.proj3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.GroupNorm(1, 512),
            nn.Linear(512, out_channels),
        )

    def forward(self, x):
        x = self.proj1(x)
        x = self.proj2(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)

        x = x.view(-1, 512)
        x = self.proj3(x)
        return x


class LAME(nn.Module):
    """LAnguage Multi-Modal Embedded."""

    def __init__(
        self,
        llm_id: str,
        vision_tower: Any,
        image_projector: nn.Module,
        audio_model_id: Union[str, None],
        max_length: int = 1024,
        **kwargs
    ) -> None:
        super().__init__()

        self.llm_id = llm_id
        self.vision_tower = vision_tower
        self.mm_image_projector = image_projector
        self.max_length = max_length
        self.kwargs = kwargs

        self._load_llm()
        if audio_model_id is not None:
            self.load_audio_model

    def _load_llm(self):
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_id,
            trust_remote_code=True,
            torch_dtype="auto",
            max_length=self.max_length,
            **self.kwargs
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_id, trust_remote_code=True)

    def _load_audio_model(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vision_tower(x)
        x = self.mm_image_projector(x)
        x = self.llm(x)

        return x
