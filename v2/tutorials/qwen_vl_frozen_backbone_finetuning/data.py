from pathlib import Path

import torch

from config import CLASS_NAMES, Config, DatasetArtifacts, dataset_env
from utils import build_occlusion_box


@dataset_env.task
async def prepare_occlusion_dataset(config: Config) -> DatasetArtifacts:
    from PIL import Image
    from torchvision.datasets import CIFAR10
    from flyte.io import Dir
    from flyteplugins.jsonl import JsonlFile
    import random

    rng = random.Random(config.seed)

    images_dir = Path("/tmp/qwen_vl_occlusion_images")
    train_images_dir = images_dir / "train" / "images"
    val_images_dir = images_dir / "validation" / "images"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        "The image may be partially occluded. "
        "Answer with exactly one CIFAR-10 class label: "
        + ", ".join(CLASS_NAMES)
        + ". What is the main object?"
    )

    async def export_split(
        dataset,
        split_name: str,
        limit: int,
        local_image_dir: Path,
        occ_min: float,
        occ_max: float,
    ):
        out = JsonlFile.new_remote(f"{split_name}_manifest.jsonl")
        async with out.writer() as writer:
            for idx in range(limit):
                pil_image, label_idx = dataset[idx]
                resized = pil_image.resize(
                    (config.image_size, config.image_size),
                    resample=Image.Resampling.BICUBIC,
                )
                rel_path = f"{split_name}/images/{split_name}-{idx:05d}.png"
                resized.save(local_image_dir / f"{split_name}-{idx:05d}.png")
                occlusion = build_occlusion_box(
                    width=config.image_size,
                    height=config.image_size,
                    rng=rng,
                    min_fraction=occ_min,
                    max_fraction=occ_max,
                )
                await writer.write(
                    {
                        "image_path": rel_path,
                        "label": CLASS_NAMES[label_idx],
                        "label_index": int(label_idx),
                        "prompt": prompt,
                        "occlusion": occlusion,
                    }
                )
        return out

    train_dataset = CIFAR10(root="/tmp/cifar10", train=True, download=True)
    val_dataset = CIFAR10(root="/tmp/cifar10", train=False, download=True)

    train_manifest = await export_split(
        train_dataset,
        "train",
        config.max_train_samples,
        train_images_dir,
        config.train_occlusion_min,
        config.train_occlusion_max,
    )
    val_manifest = await export_split(
        val_dataset,
        "validation",
        config.max_val_samples,
        val_images_dir,
        config.eval_occlusion_min,
        config.eval_occlusion_max,
    )

    return DatasetArtifacts(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        images=await Dir.from_local(str(images_dir)),
    )


class QwenOcclusionDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], images_root: Path):
        self.records = records
        self.images_root = images_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image, ImageDraw

        record = self.records[index]

        clean_image = Image.open(self.images_root / record["image_path"]).convert("RGB")
        occluded_image = clean_image.copy()

        mask = Image.new("L", clean_image.size, color=0)
        draw_image = ImageDraw.Draw(occluded_image)
        draw_mask = ImageDraw.Draw(mask)

        box = record["occlusion"]
        rectangle = [box["x0"], box["y0"], box["x1"], box["y1"]]
        draw_image.rectangle(rectangle, fill=(0, 0, 0))
        draw_mask.rectangle(rectangle, fill=255)

        return {
            "image": occluded_image,
            "clean_image": clean_image,
            "clean_image_path": record["image_path"],
            "label": record["label"],
            "prompt": record["prompt"],
            "mask_image": mask,
            "occlusion": record["occlusion"],
        }


class QwenCollator:
    def __init__(self, model_name: str, max_length: int, image_size: int):
        from transformers import AutoProcessor
        from torchvision.transforms.functional import pil_to_tensor

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=image_size * image_size,
            max_pixels=image_size * image_size,
        )
        self.image_processor = self.processor.image_processor
        self.max_length = max_length
        self.pil_to_tensor = pil_to_tensor

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        prompt_texts, full_texts, images, clean_images, masks = [], [], [], [], []
        for item in batch:
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["prompt"]},
                    ],
                }
            ]
            full_messages = user_messages + [
                {"role": "assistant", "content": item["label"]}
            ]
            prompt_texts.append(
                self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images.append(item["image"])
            clean_images.append(item["clean_image"])
            masks.append(self.pil_to_tensor(item["mask_image"]).float() / 255.0)

        full_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        clean_image_inputs = self.image_processor(
            images=clean_images, return_tensors="pt"
        )

        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)

        for row_idx, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row_idx, :prompt_length] = -100
            labels[row_idx, full_inputs["attention_mask"][row_idx] == 0] = -100

        batch_dict = {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "pixel_values": full_inputs["pixel_values"],
            "clean_pixel_values": clean_image_inputs["pixel_values"],
            "labels": labels,
            "occlusion_mask": torch.stack(masks, dim=0),
        }
        if "image_grid_thw" in full_inputs:
            batch_dict["image_grid_thw"] = full_inputs["image_grid_thw"]

        return batch_dict


def build_data_loaders(
    train_records: list[dict],
    val_records: list[dict],
    images_root: Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
    max_length: int,
    image_size: int,
):
    train_dataset = QwenOcclusionDataset(train_records, images_root)
    val_dataset = QwenOcclusionDataset(val_records, images_root)
    collator = QwenCollator(
        model_name=model_name,
        max_length=max_length,
        image_size=image_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )

    return train_loader, val_loader
