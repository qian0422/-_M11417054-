"""
run_clip_zeroshot.py

Reproduce CLIP zero-shot image classification on CIFAR-100/CIFAR-10/STL10
or a custom ImageFolder dataset.

Examples:
    python run_clip_zeroshot.py --dataset cifar100 --model ViT-B/32 --batch-size 64 --max-samples 1000
    python run_clip_zeroshot.py --dataset cifar100 --model ViT-B/32 --batch-size 64 --max-samples 0
    python run_clip_zeroshot.py --dataset cifar10 --model ViT-B/32 --batch-size 64 --max-samples 0
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from tqdm import tqdm

try:
    import clip
except ImportError as exc:
    raise SystemExit(
        "Cannot import CLIP. Please install it first:\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    ) from exc


PROMPT_TEMPLATES = {
    "basic": [
        "a photo of a {}.",
    ],
    "object": [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a close-up photo of a {}.",
        "a clean photo of a {}.",
    ],
    "scene": [
        "a photo of a {}.",
        "a photo of the {} scene.",
        "a landscape photo of {}.",
        "an image of {}.",
    ],
}


@dataclass
class EvalResult:
    dataset: str
    model: str
    prompt_type: str
    num_samples: int
    num_classes: int
    top1: float
    top5: float
    device: str
    created_at: str


class IndexedDataset(Dataset):
    """Wrap a dataset to also return the original sample index."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        return image, label, index


def clean_class_name(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ").strip()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_dataset(dataset_name: str, data_dir: str, preprocess) -> Tuple[Dataset, List[str]]:
    dataset_name = dataset_name.lower()
    root = Path(data_dir)

    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=root, train=False, transform=preprocess, download=True)
        class_names = list(dataset.classes)
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=root, train=False, transform=preprocess, download=True)
        class_names = list(dataset.classes)
    elif dataset_name == "stl10":
        dataset = datasets.STL10(root=root, split="test", transform=preprocess, download=True)
        class_names = list(dataset.classes)
    elif dataset_name == "imagefolder":
        if not root.exists():
            raise FileNotFoundError(
                f"ImageFolder data directory does not exist: {root}\n"
                "Expected format: data_dir/class_name/image.jpg"
            )
        dataset = datasets.ImageFolder(root=root, transform=preprocess)
        class_names = list(dataset.classes)
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "Choose from: cifar100, cifar10, stl10, imagefolder."
        )

    return dataset, class_names


def maybe_subset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


@torch.no_grad()
def build_text_features(model, class_names: Sequence[str], templates: Sequence[str], device: torch.device) -> torch.Tensor:
    text_features = []

    for class_name in class_names:
        label_text = clean_class_name(class_name)
        prompts = [template.format(label_text) for template in templates]
        tokens = clip.tokenize(prompts).to(device)

        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

        # Average multiple prompt embeddings, then normalize again.
        features = features.mean(dim=0)
        features = features / features.norm()
        text_features.append(features)

    return torch.stack(text_features, dim=0)


def write_confusion_matrix_csv(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """Write a class-by-class confusion matrix without requiring sklearn."""
    n_classes = len(class_names)
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for true_id, pred_id in zip(y_true, y_pred):
        matrix[int(true_id)][int(pred_id)] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(class_names))
        for i, row in enumerate(matrix):
            writer.writerow([class_names[i]] + row)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    text_features: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device,
    prediction_csv_path: Path,
    confusion_csv_path: Path,
) -> Tuple[int, float, float]:
    top1_correct = 0
    top5_correct = 0
    total = 0
    k = min(5, len(class_names))
    all_true: List[int] = []
    all_pred: List[int] = []

    prediction_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with prediction_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_label_id", "true_class", "pred_label_id", "pred_class", "correct_top1"])

        for images, labels, indices in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ text_features.T
            topk_indices = logits.topk(k, dim=1).indices

            pred1 = topk_indices[:, 0]
            top1_correct += (pred1 == labels).sum().item()
            top5_correct += (topk_indices == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

            labels_cpu = labels.cpu().tolist()
            pred_cpu = pred1.cpu().tolist()
            all_true.extend(labels_cpu)
            all_pred.extend(pred_cpu)

            for sample_index, y_true, y_pred in zip(indices.tolist(), labels_cpu, pred_cpu):
                writer.writerow([
                    sample_index,
                    y_true,
                    class_names[y_true],
                    y_pred,
                    class_names[y_pred],
                    int(y_true == y_pred),
                ])

    write_confusion_matrix_csv(all_true, all_pred, class_names, confusion_csv_path)
    top1 = top1_correct / total if total else 0.0
    top5 = top5_correct / total if total else 0.0
    return total, top1, top5


def append_metrics(result: EvalResult, metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": result.created_at,
        "dataset": result.dataset,
        "model": result.model,
        "prompt_type": result.prompt_type,
        "num_samples": result.num_samples,
        "num_classes": result.num_classes,
        "top1_accuracy": result.top1,
        "top5_accuracy": result.top5,
        "top1_percent": round(result.top1 * 100, 2),
        "top5_percent": round(result.top5 * 100, 2),
        "device": result.device,
    }

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP zero-shot reproduction script")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10", "stl10", "imagefolder"])
    parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name, e.g., ViT-B/32 or RN50")
    parser.add_argument("--prompt-type", default="basic", choices=list(PROMPT_TEMPLATES.keys()))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=1000, help="0 means full dataset")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows to avoid multiprocessing issues")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    print("=" * 72)
    print("CLIP zero-shot reproduction")
    print(f"Dataset     : {args.dataset}")
    print(f"Model       : {args.model}")
    print(f"Prompt type : {args.prompt_type}")
    print(f"Device      : {device}")
    print(f"Max samples : {args.max_samples if args.max_samples > 0 else 'full dataset'}")
    print("=" * 72)

    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    base_dataset, class_names = load_dataset(args.dataset, args.data_dir, preprocess)
    eval_dataset = maybe_subset(base_dataset, args.max_samples)
    indexed_dataset = IndexedDataset(eval_dataset)

    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    templates = PROMPT_TEMPLATES[args.prompt_type]
    text_features = build_text_features(model, class_names, templates, device)

    safe_model_name = args.model.replace("/", "-")
    output_dir = Path(args.output_dir)
    prediction_csv = output_dir / f"predictions_{args.dataset}_{safe_model_name}_{args.prompt_type}.csv"
    confusion_csv = output_dir / f"confusion_matrix_{args.dataset}_{safe_model_name}_{args.prompt_type}.csv"
    metrics_csv = output_dir / "metrics.csv"

    total, top1, top5 = evaluate(
        model=model,
        loader=loader,
        text_features=text_features,
        class_names=class_names,
        device=device,
        prediction_csv_path=prediction_csv,
        confusion_csv_path=confusion_csv,
    )

    result = EvalResult(
        dataset=args.dataset,
        model=args.model,
        prompt_type=args.prompt_type,
        num_samples=total,
        num_classes=len(class_names),
        top1=top1,
        top5=top5,
        device=str(device),
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    append_metrics(result, metrics_csv)

    print("\nFinished.")
    print(f"Samples      : {total}")
    print(f"Classes      : {len(class_names)}")
    print(f"Top-1 Acc    : {top1 * 100:.2f}%")
    print(f"Top-5 Acc    : {top5 * 100:.2f}%")
    print(f"Metrics CSV  : {metrics_csv}")
    print(f"Predictions  : {prediction_csv}")
    print(f"Confusion Mat: {confusion_csv}")


if __name__ == "__main__":
    main()
