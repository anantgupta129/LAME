import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class COCOCaptionsDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, max_length) -> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        img0 = data_dict["image"]
        img0 = expand2square(img0, tuple(int(x * 255) for x in self.processor.image_mean))
        img0 = img0.convert("RGB")
        img0 = self.image_processor(images=img0, return_tensors="pt")["pixel_values"][0]

        caption = data_dict["raw"]
        caption = self.tokenizer(
            caption, padding="max_length", truncation=True, max_length=self.max_length
        )

        return {"image": img0, "output": caption["input_ids"]}


def build_dataloader(args, image_processor, tokenizer):
    train_ds = load_dataset(args.dataset_name, split="train")
    val_ds = load_dataset(args.dataset_name, split="validation")

    trainset = COCOCaptionsDataset(
        dataset=train_ds,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    validationset = COCOCaptionsDataset(
        dataset=val_ds,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )
    val_dataloader = torch.utils.data.DataLoader(
        validationset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return train_dataloader, val_dataloader
