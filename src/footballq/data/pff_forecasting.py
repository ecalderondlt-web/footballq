"""Leakage-controlled multi-horizon forecasting targets for PFF tracking shards."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from footballq.data.td_jepa_dataset import TDJEPAData, load_td_jepa_data
from footballq.repro.splits import load_split_manifest, stable_json_bytes

PFF_FORECAST_VERSION = 1
DEFAULT_HORIZONS_SECONDS = (0.5, 1.0, 2.0, 4.0)


def _file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_tensor(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _source_root(manifest_path: Path) -> Path:
    return manifest_path.parent.parent


def _unique_frame_rows(data: TDJEPAData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_parts = []
    xy_parts = []
    mask_parts = []
    for indices, state, mask in (
        (data.context_frame_indices, data.state_t, data.mask_t),
        (data.target_frame_indices, data.state_t_plus_delta, data.mask_t_plus_delta),
    ):
        flat_frames = indices.reshape(-1)
        if flat_frames.numel() == 0:
            continue
        order = torch.argsort(flat_frames, stable=True)
        sorted_frames = flat_frames[order]
        keep = torch.ones_like(sorted_frames, dtype=torch.bool)
        keep[1:] = sorted_frames[1:] != sorted_frames[:-1]
        selected = order[keep]
        frame_parts.append(flat_frames[selected])
        xy_parts.append(state[..., :2].reshape(-1, state.shape[2], 2)[selected])
        mask_parts.append(mask.reshape(-1, mask.shape[2])[selected])
    if not frame_parts:
        entities = int(data.state_t.shape[2])
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty((0, entities, 2), dtype=torch.float32),
            torch.empty((0, entities), dtype=torch.bool),
        )
    return torch.cat(frame_parts), torch.cat(xy_parts), torch.cat(mask_parts)


def _period_frame_lookup(
    source_root: Path,
    entries: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    frame_parts = []
    xy_parts = []
    mask_parts = []
    raw_steps: list[int] = []
    for entry in entries:
        data = load_td_jepa_data(source_root / entry["path"])
        frames, xy, mask = _unique_frame_rows(data)
        if frames.numel():
            frame_parts.append(frames)
            xy_parts.append(xy)
            mask_parts.append(mask)
        if data.context_frame_indices.shape[1] > 1:
            raw_steps.extend(
                int(value)
                for value in torch.diff(data.context_frame_indices, dim=1).reshape(-1).tolist()
                if int(value) > 0
            )
    if not frame_parts:
        raise ValueError("PFF period contains no frames from which to construct forecast targets.")
    frames = torch.cat(frame_parts)
    xy = torch.cat(xy_parts)
    mask = torch.cat(mask_parts)
    order = torch.argsort(frames, stable=True)
    frames = frames[order]
    xy = xy[order]
    mask = mask[order]
    keep = torch.ones_like(frames, dtype=torch.bool)
    keep[1:] = frames[1:] != frames[:-1]
    frames = frames[keep]
    xy = xy[keep]
    mask = mask[keep]
    if not raw_steps:
        raise ValueError("Unable to infer raw-frame spacing from PFF TD contexts.")
    raw_frame_step = int(torch.tensor(raw_steps).median().item())
    return frames, xy, mask, raw_frame_step


def _targets_for_source(
    data: TDJEPAData,
    lookup_frames: torch.Tensor,
    lookup_xy: torch.Tensor,
    lookup_mask: torch.Tensor,
    *,
    horizons_seconds: tuple[float, ...],
    raw_frame_step: int,
) -> dict[str, Any]:
    raw_fps = float(raw_frame_step) * float(data.fps)
    offsets = torch.tensor(
        [int(round(value * raw_fps)) for value in horizons_seconds],
        dtype=torch.long,
    )
    desired = data.context_frame_indices[:, -1:].long() + offsets.view(1, -1)
    locations = torch.searchsorted(lookup_frames, desired)
    in_bounds = locations < len(lookup_frames)
    safe_locations = locations.clamp(max=max(len(lookup_frames) - 1, 0))
    exact = in_bounds & (lookup_frames[safe_locations] == desired)
    future_xy = lookup_xy[safe_locations].clone()
    future_mask = lookup_mask[safe_locations].clone() & exact.unsqueeze(-1)
    context_observed = data.mask_t.any(dim=1)
    future_mask &= context_observed.unsqueeze(1)
    future_xy = future_xy.masked_fill(~future_mask.unsqueeze(-1), 0.0)
    return {
        "version": PFF_FORECAST_VERSION,
        "sample_id": list(data.sample_id),
        "target_frame_indices": desired,
        "future_xy": future_xy.float(),
        "future_mask": future_mask.bool(),
        "horizons_seconds": list(horizons_seconds),
        "raw_frame_step": raw_frame_step,
        "raw_fps": raw_fps,
    }


def prepare_pff_forecast_targets(
    source_manifest_path: str | Path,
    output_root: str | Path,
    split_manifest_path: str | Path,
    *,
    horizons_seconds: tuple[float, ...] = DEFAULT_HORIZONS_SECONDS,
    included_splits: tuple[str, ...] = ("train", "val"),
    match_ids: list[str] | None = None,
    resume: bool = True,
    confirmatory_test: bool = False,
) -> dict[str, Any]:
    """Prepare endpoint targets without reading any split absent from the source manifest."""

    source_manifest_path = Path(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    split = load_split_manifest(split_manifest_path)
    if source_manifest.get("split_manifest_sha256") != split.sha256:
        raise ValueError("Forecast source and split manifest hashes do not match.")
    if source_manifest.get("feature_view") != "position_only":
        raise ValueError("Forecast target preparation requires the position_only source view.")
    allowed_splits = set(included_splits)
    if confirmatory_test:
        if allowed_splits != {"test"}:
            raise ValueError(
                "Confirmatory forecast preparation requires the test split alone."
            )
        if match_ids is not None:
            raise ValueError(
                "Confirmatory forecast preparation forbids test-match selection."
            )
    elif not allowed_splits or not allowed_splits.issubset({"train", "val"}):
        raise ValueError("Forecast preparation only permits train and validation splits.")
    selected_matches = set(str(value) for value in match_ids) if match_ids else None
    source_entries = [
        entry
        for entry in source_manifest["shards"]
        if entry["split"] in allowed_splits
        and (selected_matches is None or str(entry["match_id"]) in selected_matches)
    ]
    if not source_entries:
        raise ValueError("No source forecast shards matched the requested train/validation scope.")
    if any(entry["split"] == "test" for entry in source_entries) and not confirmatory_test:
        raise ValueError("PFF test shards are forbidden during forecast target preparation.")
    if confirmatory_test:
        selected_test_ids = {str(entry["match_id"]) for entry in source_entries}
        expected_test_ids = set(split.test_match_ids)
        if selected_test_ids != expected_test_ids:
            raise ValueError(
                "Confirmatory forecast preparation requires every frozen test match: "
                f"expected {sorted(expected_test_ids)}, got {sorted(selected_test_ids)}."
            )

    source_root = _source_root(source_manifest_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in source_entries:
        grouped[(entry["split"], str(entry["match_id"]), int(entry["period"]))].append(entry)

    output_entries: list[dict[str, Any]] = []
    for (split_name, match_id, period), entries in sorted(grouped.items()):
        entries.sort(key=lambda value: value["path"])
        lookup_frames, lookup_xy, lookup_mask, raw_frame_step = _period_frame_lookup(
            source_root, entries
        )
        for entry in entries:
            source_path = source_root / entry["path"]
            source_data = load_td_jepa_data(source_path)
            destination = (
                output_root
                / split_name
                / match_id
                / Path(entry["path"]).name.replace("td_", "forecast_")
            )
            resume_key_payload = {
                "version": PFF_FORECAST_VERSION,
                "source_tensor_sha256": entry["tensor_sha256"],
                "source_manifest_sha256": source_manifest["manifest_payload_sha256"],
                "split_manifest_sha256": split.sha256,
                "horizons_seconds": list(horizons_seconds),
            }
            resume_key = _payload_sha256(resume_key_payload)
            sidecar_path = destination.with_suffix(".json")
            if resume and destination.exists() and sidecar_path.exists():
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if sidecar.get("resume_key") == resume_key:
                    output_entries.append(sidecar)
                    continue
            targets = _targets_for_source(
                source_data,
                lookup_frames,
                lookup_xy,
                lookup_mask,
                horizons_seconds=horizons_seconds,
                raw_frame_step=raw_frame_step,
            )
            _save_tensor(destination, targets)
            sidecar = {
                "status": "complete",
                "version": PFF_FORECAST_VERSION,
                "path": str(destination.relative_to(output_root)),
                "source_path": entry["path"],
                "source_tensor_sha256": entry["tensor_sha256"],
                "target_tensor_sha256": _file_sha256(destination),
                "match_id": match_id,
                "split": split_name,
                "period": period,
                "example_count": len(source_data.sample_id),
                "valid_endpoint_count": int(targets["future_mask"].sum().item()),
                "resume_key": resume_key,
            }
            sidecar["manifest_payload_sha256"] = _payload_sha256(sidecar)
            _write_json(sidecar_path, sidecar)
            output_entries.append(sidecar)

    included_match_ids = sorted({entry["match_id"] for entry in output_entries})
    manifest = {
        "status": "complete",
        "version": PFF_FORECAST_VERSION,
        "dataset": "pff_fc",
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": source_manifest["manifest_payload_sha256"],
        "split_manifest_path": str(Path(split_manifest_path)),
        "split_manifest_sha256": split.sha256,
        "feature_view": "position_only",
        "visibility_mode": "observed_only",
        "horizons_seconds": list(horizons_seconds),
        "included_splits": sorted(allowed_splits),
        "test_included": "test" in allowed_splits,
        "access_protocol": (
            "confirmatory_test_only_v1"
            if confirmatory_test
            else "train_validation_only_v1"
        ),
        "included_match_ids": included_match_ids,
        "included_match_count": len(included_match_ids),
        "example_count": sum(int(entry["example_count"]) for entry in output_entries),
        "valid_endpoint_count": sum(
            int(entry["valid_endpoint_count"]) for entry in output_entries
        ),
        "tensor_hashes_complete": True,
        "shards": output_entries,
    }
    manifest["manifest_payload_sha256"] = _payload_sha256(manifest)
    _write_json(output_root / "dataset_manifest.json", manifest)
    return manifest


class PFFForecastDataset(Dataset):
    """Lazy paired access to source contexts and prepared forecast targets."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        cache_size: int = 1,
        verify_hashes_on_load: bool = False,
        allow_confirmatory_test: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if split == "test":
            if not allow_confirmatory_test:
                raise ValueError(
                    "PFF test forecast access requires the explicit confirmatory-test path."
                )
            if (
                self.manifest.get("test_included") is not True
                or self.manifest.get("included_splits") != ["test"]
                or self.manifest.get("access_protocol") != "confirmatory_test_only_v1"
            ):
                raise ValueError(
                    "Confirmatory PFF test access requires a test-only forecast manifest."
                )
        elif split not in {"train", "val"}:
            raise ValueError("PFF forecast datasets only permit train or validation access.")
        elif self.manifest.get("test_included"):
            raise ValueError("Forecast manifest unexpectedly includes PFF test data.")
        self.split = split
        self.root = self.manifest_path.parent
        self.source_manifest_path = Path(self.manifest["source_manifest_path"])
        self.source_root = _source_root(self.source_manifest_path)
        self.shards = [entry for entry in self.manifest["shards"] if entry["split"] == split]
        if not self.shards:
            raise ValueError(f"Forecast manifest contains no {split!r} shards.")
        self.cache_size = max(1, int(cache_size))
        self.verify_hashes_on_load = bool(verify_hashes_on_load)
        self.shard_starts = []
        self.shard_ends = []
        offset = 0
        for entry in self.shards:
            self.shard_starts.append(offset)
            offset += int(entry["example_count"])
            self.shard_ends.append(offset)
        self.example_count = offset
        self._cache: OrderedDict[int, tuple[TDJEPAData, dict[str, Any]]] = OrderedDict()
        self.prototype = self._load_shard(0)[0]

    def __len__(self) -> int:
        return self.example_count

    def _load_shard(self, shard_index: int) -> tuple[TDJEPAData, dict[str, Any]]:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        entry = self.shards[shard_index]
        source_path = self.source_root / entry["source_path"]
        target_path = self.root / entry["path"]
        if self.verify_hashes_on_load:
            if _file_sha256(source_path) != entry["source_tensor_sha256"]:
                raise ValueError(f"Forecast source tensor hash mismatch: {source_path}")
            if _file_sha256(target_path) != entry["target_tensor_sha256"]:
                raise ValueError(f"Forecast target tensor hash mismatch: {target_path}")
        source = load_td_jepa_data(source_path)
        target = torch.load(target_path, map_location="cpu", weights_only=False)
        if source.sample_id != list(target["sample_id"]):
            raise ValueError(f"Forecast source/target sample identities differ: {target_path}")
        if len(source.sample_id) != int(entry["example_count"]):
            raise ValueError(f"Forecast shard count does not match manifest: {target_path}")
        value = (source, target)
        self._cache[shard_index] = value
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.shard_ends, index)
        local_index = index - self.shard_starts[shard_index]
        source, target = self._load_shard(shard_index)
        return {
            "state_t": source.state_t[local_index],
            "mask_t": source.mask_t[local_index],
            "future_xy": target["future_xy"][local_index],
            "future_mask": target["future_mask"][local_index],
            "entity_type": source.entity_type[local_index],
            "team_id": source.team_id[local_index],
            "sample_id": source.sample_id[local_index],
            "match_id": source.match_id[local_index],
            "period": source.period[local_index],
            "frame_t": source.frame_t[local_index],
            "target_frame_indices": target["target_frame_indices"][local_index],
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state
