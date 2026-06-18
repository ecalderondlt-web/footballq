"""Export TD-JEPA state embeddings for later probe experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from footballq.data.td_jepa_dataset import TDJEPADataset
from footballq.training.eval_td_jepa import load_td_checkpoint_model
from footballq.training.train_td_jepa import td_batch_to_device


def export_td_embeddings(
    checkpoint: str | Path,
    data_path: str | Path,
    out: str | Path,
    split: str = "test",
    device: str | None = "auto",
) -> Path:
    model, data, cfg, payload, torch_device = load_td_checkpoint_model(
        checkpoint,
        data_path=data_path,
        device=device,
    )
    split_indices = payload.get("split_indices", {})
    indices = split_indices.get(split, list(range(len(data.match_id))))
    loader = DataLoader(
        TDJEPADataset(data, indices=indices),
        batch_size=int(cfg.get("training", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("training", {}).get("num_workers", 0)),
    )
    z_parts: list[torch.Tensor] = []
    match_ids: list[str] = []
    frame_ts: list[int] = []
    delta_frames: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            device_batch = td_batch_to_device(batch, torch_device)
            z = model.online_encoder(device_batch["state_t"], device_batch["mask_t"])
            z_parts.append(z.detach().cpu())
            match_ids.extend(str(value) for value in batch["match_id"])
            frame_ts.extend(int(value) for value in batch["frame_t"])
            delta_frames.extend(int(value) for value in batch["delta_frames"])
    z_all = torch.cat(z_parts, dim=0) if z_parts else torch.empty((0, cfg["model"]["z_dim"]))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_out: dict[str, Any] = {
        "z": z_all,
        "match_id": match_ids,
        "frame_t": frame_ts,
        "delta_frames": delta_frames,
        "source_split": split,
        "config": cfg,
    }
    torch.save(payload_out, out_path)
    return out_path
