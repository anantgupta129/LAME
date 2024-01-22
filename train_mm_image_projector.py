from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from dataset import build_dataloader
from model import build_lame


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    llm_model_id: Optional[str] = field(default="microsoft/phi-2")
    vision_model_id: Optional[str] = field(default="wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M")
    audio_model_id: Optional[str] = None
    mm_vision_select_layer: int = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_image_projection_out_size: int = field(default=1024)


@dataclass
class DataArguments:
    dataset_name: str = "HuggingFaceM4/COCO"
    batch_size: int = 256
    captions_max_length: int = field(default=1024)


@dataclass
class TrainingArguments:
    optim: str = field(default="adamw")
    save_dir: Optional[str] = field(default="checkpoints")
    num_epochs: int = field(default=1)
    lr: float = field(default=1e-4)
    weight_decay: float = field(default=0.001)
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=8, metadata={"help": "How many bits to use."})
    num_workers: int = field(default=4)
    pin_memory: bool = field(default=True)
    scheduler: str = field(default="onecycle")
    project_name: Optional[str] = field(default="LAME (LAnguage Multi-Modal Embedded)")
    experiment_name: Optional[str] = field(default="train_mm_image_projection")


def train_one_epoch(
    model: nn.Module,
    optimizer: optim,
    scheduler: optim.lr_scheduler,
    loader: DataLoader,
    criterion: nn.Module,
    args: Any,
):
    model.train()

    for idx, data in enumerate(tqdm(loader)):
        images = data["image"].to(args.device)
        target = data["target"].to(args.device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        wandb.log({"train_loss": loss.item()}, step=idx)

        tqdm.set_postfix(loss=loss.item())


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, args: Any
) -> torch.Tensor:
    model.eval()

    losses = torch.zeros(len(loader), device=args.device)
    with torch.no_grad():
        for idx, data in enumerate(loader):
            images = data["image"].to(args.device)
            target = data["target"].to(args.device)

            logits = model(images)
            loss = criterion(logits, target)
            losses[idx] = loss

            wandb.log({"val_loss": loss.item()}, step=idx)

    loss = torch.mean(losses)
    print("[-] Validation loss: ", loss.item())

    return loss


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    args = parser.parse_args()
    wdir = Path(args.save_dir) / "mm_image_projection"

    wdir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=args.project_name, name=args.experiment_name)
    wandb.config.update(args)

    # compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    compute_dtype = torch.float32
    bnb_model_from_pretrained_args = {}
    if args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": args.device},
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=args.bits == 4,
                    load_in_8bit=args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=args.double_quant,
                    bnb_4bit_quant_type=args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model, mm_image_processor, mm_tokenizer = build_lame(args, **bnb_model_from_pretrained_args)
    train_dataloader, val_dataloader = build_dataloader(args, mm_image_processor, mm_tokenizer)
    criterion = nn.CrossEntropyLoss()
    model.llm.requires_grad_(False)

    if args.optim == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(
            f"Unsupported optimizer: {args.optim}, add or select from ['adamw', 'adam']"
        )

    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=len(train_dataloader),
            epochs=args.num_epochs,
            pct_start=5 / args.num_epochs,
            div_factor=100,
            final_div_factor=100,
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_dataloader), eta_min=1e-5
        )

    best_loss = torch.inf
    for epoch in range(args.num_epochs):
        print(f"[x] Epoch {epoch}/{args.num_epochs}")
        train_one_epoch(model, optimizer, scheduler, train_dataloader, criterion, args)
        val_loss = evaluate_one_epoch(model, val_dataloader, criterion, args)
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"[x] Best model saved (loss: {best_loss}) ")
            torch.save(model.mm_image_projector.state_dict(), wdir / "mm_image_projection.pth")


if __name__ == "__main__":
    train()
