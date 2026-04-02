import base64
import html
import io
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from flyte.io import Dir, File
from flyteplugins.jsonl import JsonlFile

from config import Config, DEVICES_PER_NODE, NUM_NODES


def load_jsonl_records(jf: JsonlFile) -> list[dict]:
    return [record for record in jf.iter_records_sync()]


async def load_jsonl_records_async(jf: JsonlFile) -> list[dict]:
    records = []
    async for record in jf.iter_records():
        records.append(record)
    return records


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace()).strip()


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(
        param.numel() for param in module.parameters() if param.requires_grad
    )
    return total, trainable


def upload_files_to_uri(
    root_dir: Path,
    base_uri: str,
    relative_paths: list[Path],
) -> list[str]:
    uploaded_paths = []
    for relative_path in relative_paths:
        local_path = root_dir / relative_path
        if not local_path.is_file():
            continue
        remote_path = f"{base_uri.rstrip('/')}/{relative_path.as_posix()}"
        uploaded = File.from_local_sync(str(local_path), remote_path)
        uploaded_paths.append(uploaded.path)
    return uploaded_paths


def build_occlusion_box(
    width: int,
    height: int,
    rng: random.Random,
    min_fraction: float,
    max_fraction: float,
) -> dict:
    occ_w = max(8, int(width * rng.uniform(min_fraction, max_fraction)))
    occ_h = max(8, int(height * rng.uniform(min_fraction, max_fraction)))
    x0 = rng.randint(0, max(0, width - occ_w))
    y0 = rng.randint(0, max(0, height - occ_h))
    return {"x0": x0, "y0": y0, "x1": x0 + occ_w, "y1": y0 + occ_h}


def download_dir_sync(dataset_dir: Dir) -> Path:
    return Path(dataset_dir.download_sync())


async def download_dir_async(dataset_dir: Dir) -> Path:
    return Path(await dataset_dir.download())


def find_resume_checkpoint(artifacts_root: Path) -> Optional[Path]:
    checkpoints_dir = artifacts_root / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    last_checkpoint = checkpoints_dir / "latest.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint
    checkpoint_paths = list(checkpoints_dir.glob("*.ckpt"))
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=lambda path: path.stat().st_mtime)


def build_recovery_uri(base_uri: Optional[str], run_name: str) -> Optional[str]:
    if not base_uri:
        return None
    return f"{base_uri.rstrip('/')}/{run_name}/qwen_vl_training_recovery"


def write_training_artifacts(
    output_dir: Path,
    *,
    config: Config,
    module: "QwenVLAdapterModule",
    metrics_history: list[dict],
    world_size: int,
    grad_accum_steps: int,
    best_model_path: str,
    last_model_path: str,
    resumed_from: Optional[str],
    status: str,
    error_message: Optional[str] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_history, indent=2),
        encoding="utf-8",
    )

    summary = {
        "model_name": config.model_name,
        "world_size": world_size,
        "num_nodes": NUM_NODES,
        "devices_per_node": DEVICES_PER_NODE,
        "per_device_batch_size": config.per_device_batch_size,
        "target_global_batch_size": config.target_global_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "reconstruction_loss_weight": config.reconstruction_loss_weight,
        "total_params": module.total_params,
        "trainable_params": module.trainable_params,
        "trainable_fraction": module.trainable_params / max(1, module.total_params),
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
        "resumed_from": resumed_from,
        "status": status,
        "error_message": error_message,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def image_to_base64(path: str) -> str:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_loss_curve_svg(metrics: list[dict]) -> str:
    if not metrics:
        return (
            "<div style='padding:16px; border:1px dashed #bbb; border-radius:12px;'>"
            "No training history is available yet."
            "</div>"
        )

    width = 920
    height = 260
    left = 56
    right = 18
    top = 18
    bottom = 36
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = []
    for row in metrics:
        values.extend(
            [
                float(row.get("train_loss", 0.0)),
                float(row.get("val_loss", 0.0)),
                float(row.get("train_reconstruction_loss", 0.0)),
                float(row.get("val_reconstruction_loss", 0.0)),
            ]
        )
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    epochs = [int(row["epoch"]) for row in metrics]
    x_min = min(epochs)
    x_max = max(epochs)

    def point(epoch: int, value: float, index: int) -> str:
        if x_max == x_min:
            x = left + plot_width * 0.5
        else:
            x = left + ((epoch - x_min) / (x_max - x_min)) * plot_width
        y = top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_height
        if len(metrics) == 1:
            x = left + plot_width * 0.5
        if index == 0 and len(metrics) > 1:
            x = left
        if index == len(metrics) - 1 and len(metrics) > 1:
            x = left + plot_width
        return f"{x:.1f},{y:.1f}"

    def polyline(metric_key: str) -> str:
        return " ".join(
            point(int(row["epoch"]), float(row.get(metric_key, 0.0)), idx)
            for idx, row in enumerate(metrics)
        )

    x_ticks = "".join(
        (
            f"<text x='{left + (idx / max(1, len(metrics) - 1)) * plot_width:.1f}' "
            f"y='{height - 10}' font-size='11' text-anchor='middle' fill='#6b7280'>"
            f"{epoch}</text>"
        )
        for idx, epoch in enumerate(epochs)
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" style="width:100%; max-width:920px; background:#fff; border:1px solid #d7dce2; border-radius:12px;">
        <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#9aa4b2" stroke-width="1.5" />
        <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#9aa4b2" stroke-width="1.5" />
        <text x="12" y="{top + 6}" font-size="11" fill="#6b7280">{y_max:.3f}</text>
        <text x="12" y="{top + plot_height}" font-size="11" fill="#6b7280">{y_min:.3f}</text>
        <polyline points="{polyline('train_loss')}" fill="none" stroke="#0f766e" stroke-width="3" />
        <polyline points="{polyline('val_loss')}" fill="none" stroke="#1d4ed8" stroke-width="3" />
        <polyline points="{polyline('train_reconstruction_loss')}" fill="none" stroke="#d97706" stroke-width="2" stroke-dasharray="6 4" />
        <polyline points="{polyline('val_reconstruction_loss')}" fill="none" stroke="#b91c1c" stroke-width="2" stroke-dasharray="6 4" />
        {x_ticks}
        <text x="{left + plot_width / 2:.1f}" y="{height - 10}" font-size="11" text-anchor="middle" fill="#6b7280">Epoch</text>
    </svg>
    """


def build_qwen_adapter_report_html(
    training_root: Path,
    evaluation_root: Optional[Path] = None,
) -> str:
    summary = json.loads((training_root / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((training_root / "metrics.json").read_text(encoding="utf-8"))

    predictions = []
    eval_summary = {"evaluated_examples": 0, "exact_match": None}
    if evaluation_root is not None:
        predictions = json.loads(
            (evaluation_root / "predictions.json").read_text(encoding="utf-8")
        )
        eval_summary = json.loads(
            (evaluation_root / "evaluation_summary.json").read_text(encoding="utf-8")
        )

    metrics_rows = "".join(
        (
            "<tr>"
            f"<td style='padding:8px;'>{row['epoch']}</td>"
            f"<td style='padding:8px;'>{row['train_loss']:.4f}</td>"
            f"<td style='padding:8px;'>{row['val_loss']:.4f}</td>"
            f"<td style='padding:8px;'>{row.get('train_reconstruction_loss', 0.0):.4f}</td>"
            f"<td style='padding:8px;'>{row.get('val_reconstruction_loss', 0.0):.4f}</td>"
            f"<td style='padding:8px;'>{row['adapter_gate']:.4f}</td>"
            "</tr>"
        )
        for row in metrics
    )

    sample_cards = []
    for prediction in predictions[:4]:
        encoded = image_to_base64(
            str(evaluation_root / prediction["occluded_image_path"])
        )
        border = "#166534" if prediction["match"] else "#991b1b"
        sample_cards.append(
            f"""
            <div style="border:1px solid {border}; border-radius:12px; padding:12px; width:260px;">
                <img src="data:image/png;base64,{encoded}" style="width:100%; border-radius:8px;" />
                <div style="margin-top:10px; font-size:14px; line-height:1.45;">
                    <div><strong>Label:</strong> {html.escape(prediction['label'])}</div>
                    <div><strong>Prediction:</strong> {html.escape(prediction['prediction'])}</div>
                    <div><strong>Match:</strong> {prediction['match']}</div>
                </div>
            </div>
            """
        )
    sample_cards_html = (
        "".join(sample_cards)
        if sample_cards
        else (
            "<div style='padding:12px; border:1px dashed #bbb; border-radius:12px;'>"
            "No evaluation samples are available for this run."
            "</div>"
        )
    )

    eval_exact_match_html = (
        f"{eval_summary['exact_match']:.2%}"
        if eval_summary["exact_match"] is not None
        else "N/A"
    )
    status_color = "#166534" if summary.get("status") == "completed" else "#991b1b"
    loss_curve_svg = _build_loss_curve_svg(metrics)

    return f"""
    <div style="font-family:Arial,sans-serif; padding:18px; max-width:1120px;">
        <h1 style="margin-bottom:8px;">Qwen2.5-VL Multi-Node DeepSpeed Tutorial</h1>
        <p style="font-size:15px; color:#444;">
            This run demonstrates frozen-backbone adaptation on top of
            <code>{html.escape(summary['model_name'])}</code>. The pretrained Qwen
            weights stay frozen while the new occlusion-aware Conv2d adapter learns
            restoration-aware residuals before the vision encoder.
        </p>
        <div style="margin-top:14px; padding:12px; border-radius:12px; background:#fff7f7; border:1px solid {status_color};">
            <div><strong>Status:</strong> {html.escape(summary.get('status', 'unknown'))}</div>
            <div><strong>Resumed From:</strong> {html.escape(summary.get('resumed_from') or 'fresh run')}</div>
            <div><strong>Error:</strong> {html.escape(summary.get('error_message') or 'None')}</div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; margin-top:18px;">
            <div style="background:#f5f7fa; border-radius:12px; padding:14px;">
                <div style="font-size:12px; color:#666;">Nodes x GPUs</div>
                <div style="font-size:24px; font-weight:700;">{summary['num_nodes']} x {summary['devices_per_node']}</div>
            </div>
            <div style="background:#f5f7fa; border-radius:12px; padding:14px;">
                <div style="font-size:12px; color:#666;">Trainable Params</div>
                <div style="font-size:24px; font-weight:700;">{summary['trainable_params']:,}</div>
            </div>
            <div style="background:#f5f7fa; border-radius:12px; padding:14px;">
                <div style="font-size:12px; color:#666;">Total Params</div>
                <div style="font-size:24px; font-weight:700;">{summary['total_params']:,}</div>
            </div>
            <div style="background:#f5f7fa; border-radius:12px; padding:14px;">
                <div style="font-size:12px; color:#666;">Eval Exact Match</div>
                <div style="font-size:24px; font-weight:700;">{eval_exact_match_html}</div>
            </div>
        </div>
        <h2 style="margin-top:24px;">What This Run Shows</h2>
        <ul style="line-height:1.6;">
            <li>Dataset prep is isolated as a v2 task.</li>
            <li>Training scales with Elastic plus DeepSpeed across multiple nodes.</li>
            <li>The pretrained backbone is frozen while only the inserted adapter is optimized.</li>
            <li>Evaluation renders the final report directly from task artifacts.</li>
        </ul>
        <h2 style="margin-top:24px;">Training Curves</h2>
        <div style="margin-bottom:12px; display:flex; gap:16px; flex-wrap:wrap; color:#475569; font-size:13px;">
            <div><span style="display:inline-block; width:12px; height:12px; background:#0f766e; border-radius:999px; margin-right:6px;"></span>Train total loss</div>
            <div><span style="display:inline-block; width:12px; height:12px; background:#1d4ed8; border-radius:999px; margin-right:6px;"></span>Val total loss</div>
            <div><span style="display:inline-block; width:12px; height:12px; background:#d97706; border-radius:999px; margin-right:6px;"></span>Train reconstruction loss</div>
            <div><span style="display:inline-block; width:12px; height:12px; background:#b91c1c; border-radius:999px; margin-right:6px;"></span>Val reconstruction loss</div>
        </div>
        {loss_curve_svg}
        <h2 style="margin-top:24px;">Training History</h2>
        <table style="border-collapse:collapse; width:100%;">
            <thead>
                <tr style="background:#eef2f7;">
                    <th style="text-align:left; padding:8px;">Epoch</th>
                    <th style="text-align:left; padding:8px;">Train Total</th>
                    <th style="text-align:left; padding:8px;">Val Total</th>
                    <th style="text-align:left; padding:8px;">Train Recon</th>
                    <th style="text-align:left; padding:8px;">Val Recon</th>
                    <th style="text-align:left; padding:8px;">Adapter Gate</th>
                </tr>
            </thead>
            <tbody>{metrics_rows}</tbody>
        </table>
        <h2 style="margin-top:24px;">Sample Predictions</h2>
        <div style="display:flex; flex-wrap:wrap; gap:12px;">{sample_cards_html}</div>
    </div>
    """
