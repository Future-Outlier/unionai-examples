import json
import os
import shutil
from pathlib import Path
from typing import Optional

import flyte.report
import lightning as L
import torch
from flyteplugins.wandb import get_wandb_run

from config import Config, DEVICES_PER_NODE, NUM_NODES
from utils import upload_files_to_uri, write_training_artifacts


class AdapterMetricsCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.history: list[dict] = []

    def state_dict(self) -> dict:
        return {"history": self.history}

    def load_state_dict(self, state_dict: dict) -> None:
        self.history = state_dict.get("history", self.history)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        row = {
            "epoch": int(trainer.current_epoch),
            "train_loss": float(metrics.get("train/total_loss_epoch", 0.0)),
            "train_reconstruction_loss": float(
                metrics.get("train/reconstruction_loss_epoch", 0.0)
            ),
            "train_lm_loss": float(metrics.get("train/lm_loss_epoch", 0.0)),
            "val_loss": float(metrics.get("val/total_loss", 0.0)),
            "val_reconstruction_loss": float(
                metrics.get("val/reconstruction_loss", 0.0)
            ),
            "val_lm_loss": float(metrics.get("val/lm_loss", 0.0)),
            "adapter_gate": float(torch.tanh(pl_module.adapter.gate).detach().cpu()),
        }
        if self.history and self.history[-1]["epoch"] == row["epoch"]:
            self.history[-1] = row
        else:
            self.history.append(row)

        run = get_wandb_run()
        if run and trainer.global_rank == 0:
            run.log(
                {
                    "epoch": row["epoch"],
                    "train/total_loss_epoch": row["train_loss"],
                    "train/reconstruction_loss_epoch": row["train_reconstruction_loss"],
                    "train/lm_loss_epoch": row["train_lm_loss"],
                    "val/total_loss": row["val_loss"],
                    "val/reconstruction_loss": row["val_reconstruction_loss"],
                    "val/lm_loss": row["val_lm_loss"],
                    "adapter/gate": row["adapter_gate"],
                }
            )


class LiveTrainingReportCallback(L.Callback):
    def __init__(
        self,
        *,
        report_every_n_steps: int,
        resumed_from: Optional[str],
        recovery_callback: "RecoveryArtifactCallback",
        prior_metrics: Optional[list] = None,
    ):
        super().__init__()
        self.report_every_n_steps = report_every_n_steps
        self.resumed_from = resumed_from
        self.recovery_callback = recovery_callback
        self.prior_metrics = prior_metrics or []
        self._initialized = False
        self._last_step = -1

    def on_train_start(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0 or self._initialized:
            return
        flyte.report.log(
            f"""
            <div style="font-family:Arial,sans-serif; padding:18px; max-width:980px;">
                <h1 style="margin:0 0 8px 0;">Qwen2.5-VL Live Training</h1>
                <p style="color:#444; margin:0 0 16px 0;">
                    Frozen Qwen backbone + trainable occlusion adapter on
                    {NUM_NODES} nodes x {DEVICES_PER_NODE} GPUs.
                </p>
                <div id="live-status" style="padding:12px; border-radius:12px; background:#f5f7fa; margin-bottom:14px;">
                    <div><strong>Status:</strong> running</div>
                    <div><strong>Resumed From:</strong> {self.resumed_from or 'fresh run'}</div>
                    <div><strong>Recovery Bundle:</strong> pending first checkpoint upload</div>
                </div>
                <div style="margin-bottom:14px; padding:14px; border:1px solid #d7dce2; border-radius:12px; background:#fff;">
                    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:10px; color:#475569; font-size:13px;">
                        <div><span style="display:inline-block; width:12px; height:12px; background:#0f766e; border-radius:999px; margin-right:6px;"></span>Train total loss</div>
                        <div><span style="display:inline-block; width:12px; height:12px; background:#1d4ed8; border-radius:999px; margin-right:6px;"></span>Validation total loss</div>
                    </div>
                    <svg id="live-loss-chart" viewBox="0 0 900 240" style="width:100%; height:auto;">
                        <line x1="52" y1="18" x2="52" y2="202" stroke="#9aa4b2" stroke-width="1.5"></line>
                        <line x1="52" y1="202" x2="876" y2="202" stroke="#9aa4b2" stroke-width="1.5"></line>
                        <polyline id="live-train-line" fill="none" stroke="#0f766e" stroke-width="3"></polyline>
                        <polyline id="live-val-line" fill="none" stroke="#1d4ed8" stroke-width="3"></polyline>
                        <text id="live-y-max" x="8" y="24" font-size="11" fill="#6b7280">-</text>
                        <text id="live-y-min" x="8" y="202" font-size="11" fill="#6b7280">-</text>
                        <text id="live-x-start" x="52" y="226" font-size="11" fill="#6b7280">0</text>
                        <text id="live-x-end" x="876" y="226" font-size="11" text-anchor="end" fill="#6b7280">0</text>
                        <text x="464" y="226" font-size="11" text-anchor="middle" fill="#6b7280">Global step</text>
                    </svg>
                </div>
                <table style="border-collapse:collapse; width:100%; margin-bottom:14px;">
                    <thead>
                        <tr style="background:#eef2f7;">
                            <th style="text-align:left; padding:8px;">Step</th>
                            <th style="text-align:left; padding:8px;">Phase</th>
                            <th style="text-align:left; padding:8px;">Train Total</th>
                            <th style="text-align:left; padding:8px;">Train LM</th>
                            <th style="text-align:left; padding:8px;">Train Recon</th>
                            <th style="text-align:left; padding:8px;">Val Total</th>
                            <th style="text-align:left; padding:8px;">Adapter Gate</th>
                            <th style="text-align:left; padding:8px;">Note</th>
                        </tr>
                    </thead>
                    <tbody id="live-table-body">
                        <tr id="live-placeholder-row">
                            <td colspan="8" style="padding:10px; color:#666;">Waiting for the first logged training step...</td>
                        </tr>
                    </tbody>
                </table>
                <div id="live-note" style="font-size:13px; color:#666;">
                    Waiting for the first logged training step...
                </div>
            </div>
            <script>
            window.qwenLiveHistory = [];
            window.drawQwenLiveChart = function() {{
                const history = window.qwenLiveHistory || [];
                const values = history.flatMap((point) => {{
                    const nums = [];
                    if (point.train_total_value !== null) nums.push(point.train_total_value);
                    if (point.val_total_value !== null) nums.push(point.val_total_value);
                    return nums;
                }});
                if (!values.length) return;
                let minY = Math.min(...values);
                let maxY = Math.max(...values);
                if (minY === maxY) {{
                    minY -= 1;
                    maxY += 1;
                }}
                const steps = history.map((point, index) => point.step ?? index);
                const minStep = Math.min(...steps);
                const maxStep = Math.max(...steps);
                const xFor = (step, index, total) => {{
                    if (maxStep === minStep) {{
                        return 52 + ((index + 1) / (total + 1)) * 824;
                    }}
                    return 52 + ((step - minStep) / (maxStep - minStep)) * 824;
                }};
                const yFor = (value) => 18 + (1 - ((value - minY) / (maxY - minY))) * 184;
                const toPolyline = (key) => history
                    .map((point, index) => {{
                        const value = point[key];
                        if (value === null || value === undefined) return null;
                        return `${{xFor(point.step ?? index, index, history.length).toFixed(1)}},${{yFor(value).toFixed(1)}}`;
                    }})
                    .filter(Boolean)
                    .join(" ");

                const trainLine = document.getElementById("live-train-line");
                const valLine = document.getElementById("live-val-line");
                const yMax = document.getElementById("live-y-max");
                const yMin = document.getElementById("live-y-min");
                const xStart = document.getElementById("live-x-start");
                const xEnd = document.getElementById("live-x-end");
                if (trainLine) trainLine.setAttribute("points", toPolyline("train_total_value"));
                if (valLine) valLine.setAttribute("points", toPolyline("val_total_value"));
                if (yMax) yMax.textContent = maxY.toFixed(4);
                if (yMin) yMin.textContent = minY.toFixed(4);
                if (xStart) xStart.textContent = String(minStep);
                if (xEnd) xEnd.textContent = String(maxStep);
            }};
            window.updateQwenLiveReport = function(payload) {{
                const status = document.getElementById("live-status");
                if (status) {{
                    status.innerHTML =
                        "<div><strong>Status:</strong> " + payload.status + "</div>" +
                        "<div><strong>Resumed From:</strong> " + payload.resumed_from + "</div>" +
                        "<div><strong>Recovery Bundle:</strong> " + payload.recovery_path + "</div>";
                }}
                const body = document.getElementById("live-table-body");
                if (body) {{
                    const placeholder = document.getElementById("live-placeholder-row");
                    if (placeholder) placeholder.remove();
                    const row = document.createElement("tr");
                    const values = [
                        payload.step,
                        payload.phase,
                        payload.train_total,
                        payload.train_lm,
                        payload.train_recon,
                        payload.val_total,
                        payload.adapter_gate,
                        payload.note,
                    ];
                    values.forEach((value) => {{
                        const cell = document.createElement("td");
                        cell.style.padding = "8px";
                        cell.style.borderTop = "1px solid #eef2f7";
                        cell.textContent = String(value);
                        row.appendChild(cell);
                    }});
                    body.appendChild(row);
                    while (body.rows.length > 160) {{
                        body.deleteRow(0);
                    }}
                }}
                const note = document.getElementById("live-note");
                if (note) {{
                    note.textContent = payload.note;
                }}
                if (!payload.skip_chart) {{
                    window.qwenLiveHistory.push(payload);
                    if (window.qwenLiveHistory.length > 160) {{
                        window.qwenLiveHistory.shift();
                    }}
                    window.drawQwenLiveChart();
                }}
            }};
            // Seed prior-run rows into the table for resumed runs, but keep the
            // live chart focused on the current run's global-step timeline.
            (function() {{
                const priorRows = {json.dumps(
                    [
                        {
                            "step": row["epoch"],
                            "phase": "prior/val",
                            "train_total": f"{row['train_loss']:.4f}",
                            "train_lm": f"{row.get('train_lm_loss', 0.0):.4f}",
                            "train_recon": f"{row.get('train_reconstruction_loss', 0.0):.4f}",
                            "val_total": f"{row['val_loss']:.4f}",
                            "adapter_gate": f"{row['adapter_gate']:.4f}",
                            "status": "completed",
                            "resumed_from": self.resumed_from or "fresh run",
                            "recovery_path": "-",
                            "note": f"Prior run \u2014 epoch {row['epoch']}",
                            "train_total_value": row["train_loss"],
                            "val_total_value": row["val_loss"],
                            "skip_chart": True,
                        }
                        for row in self.prior_metrics
                    ]
                )};
                priorRows.forEach(function(payload) {{
                    window.updateQwenLiveReport(payload);
                }});
            }})();
            </script>
            """,
            do_flush=True,
        )
        self._initialized = True

    # {{docs-fragment live-report-push}}
    def _push_update(
        self,
        *,
        trainer,
        pl_module,
        status: str,
        phase: str,
        train_total=None,
        train_lm=None,
        train_recon=None,
        val_total=None,
        note: str,
    ) -> None:
        adapter_gate = float(torch.tanh(pl_module.adapter.gate).detach().cpu())

        def fmt(value):
            return f"{float(value):.4f}" if value is not None else "-"

        payload = {
            "step": trainer.global_step,
            "phase": phase,
            "train_total": fmt(train_total),
            "train_lm": fmt(train_lm),
            "train_recon": fmt(train_recon),
            "val_total": fmt(val_total),
            "train_total_value": (
                float(train_total) if train_total is not None else None
            ),
            "val_total_value": float(val_total) if val_total is not None else None,
            "adapter_gate": f"{adapter_gate:.4f}",
            "status": status,
            "resumed_from": self.resumed_from or "fresh run",
            "recovery_path": self.recovery_callback.latest_path
            or "pending first checkpoint upload",
            "note": note,
        }
        flyte.report.log(
            f"""
            <script>
            if (typeof window.updateQwenLiveReport === "function") {{
                window.updateQwenLiveReport({json.dumps(payload)});
            }}
            </script>
            """,
            do_flush=True,
        )
    # {{/docs-fragment}}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        if (
            step == 0
            or step == self._last_step
            or step % self.report_every_n_steps != 0
        ):
            return

        metrics = trainer.callback_metrics
        self._push_update(
            trainer=trainer,
            pl_module=pl_module,
            status="running",
            phase="train",
            train_total=metrics.get("train/total_loss_step"),
            train_lm=metrics.get("train/lm_loss_step"),
            train_recon=metrics.get("train/reconstruction_loss_step"),
            note=f"Training step {step} logged from rank 0.",
        )
        self._last_step = step

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        metrics = trainer.callback_metrics
        self._push_update(
            trainer=trainer,
            pl_module=pl_module,
            status="running",
            phase="validation",
            train_total=metrics.get("train/total_loss_epoch"),
            train_lm=metrics.get("train/lm_loss_epoch"),
            train_recon=metrics.get("train/reconstruction_loss_epoch"),
            val_total=metrics.get("val/total_loss"),
            note=f"Validation completed for epoch {trainer.current_epoch}.",
        )


class RecoveryArtifactCallback(L.Callback):
    def __init__(
        self,
        *,
        checkpoints_dir: Path,
        output_dir: Path,
        config: Config,
        metrics_callback: AdapterMetricsCallback,
        world_size: int,
        grad_accum_steps: int,
        resumed_from: Optional[str],
        recovery_uri: Optional[str],
    ):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.output_dir = output_dir
        self.config = config
        self.metrics_callback = metrics_callback
        self.world_size = world_size
        self.grad_accum_steps = grad_accum_steps
        self.resumed_from = resumed_from
        self.recovery_uri = recovery_uri
        self.latest_path: Optional[str] = None

    def sync_artifacts(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *,
        status: str = "running",
        error_message: Optional[str] = None,
    ) -> None:
        if trainer.sanity_checking:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        checkpoint_saved = False
        recovery_dir = self.output_dir / "recovery"
        recovery_ckpt_dir = recovery_dir / "checkpoints"

        if self.recovery_uri is not None:
            recovery_ckpt_dir.mkdir(parents=True, exist_ok=True)

            try:
                trainer.save_checkpoint(str(recovery_ckpt_dir / "latest.ckpt"))
                checkpoint_saved = True
            except Exception as e:
                if trainer.global_rank == 0:
                    print(f"Warning: could not save recovery checkpoint: {e}")

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                checkpoint_saved_tensor = torch.tensor(
                    int(checkpoint_saved),
                    device=pl_module.device,
                )
                torch.distributed.all_reduce(
                    checkpoint_saved_tensor,
                    op=torch.distributed.ReduceOp.MIN,
                )
                checkpoint_saved = bool(checkpoint_saved_tensor.item())

        trainer.strategy.barrier()

        if trainer.global_rank == 0:
            write_training_artifacts(
                self.output_dir,
                config=self.config,
                module=pl_module,
                metrics_history=self.metrics_callback.history,
                world_size=self.world_size,
                grad_accum_steps=self.grad_accum_steps,
                best_model_path=getattr(
                    trainer.checkpoint_callback, "best_model_path", ""
                ),
                last_model_path=getattr(
                    trainer.checkpoint_callback, "last_model_path", ""
                ),
                resumed_from=self.resumed_from,
                status=status,
                error_message=error_message,
            )

            if self.recovery_uri is not None:
                recovery_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    self.output_dir / "metrics.json", recovery_dir / "metrics.json"
                )
                shutil.copy2(
                    self.output_dir / "summary.json", recovery_dir / "summary.json"
                )

        trainer.strategy.barrier()

        if self.recovery_uri is None or not checkpoint_saved:
            return

        checkpoint_root = recovery_ckpt_dir / "latest.ckpt"
        if local_rank == 0 and checkpoint_root.exists():
            checkpoint_files = [
                path.relative_to(recovery_dir)
                for path in checkpoint_root.rglob("*")
                if path.is_file()
            ]
            upload_files_to_uri(recovery_dir, self.recovery_uri, checkpoint_files)

        if trainer.global_rank == 0:
            upload_files_to_uri(
                recovery_dir,
                self.recovery_uri,
                [Path("metrics.json"), Path("summary.json")],
            )

        trainer.strategy.barrier()

        if trainer.global_rank == 0:
            self.latest_path = self.recovery_uri
            print(f"Recovery artifacts uploaded: {self.latest_path}")

    def on_validation_end(self, trainer, pl_module) -> None:
        self.sync_artifacts(trainer, pl_module)
