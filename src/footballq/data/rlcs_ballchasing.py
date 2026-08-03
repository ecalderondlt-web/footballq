"""Deterministic Ballchasing acquisition for the RLCS identity experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BALLCHASING_API_BASE = "https://ballchasing.com/api"
DEFAULT_ROOT_LABELS = {
    "europe-bizraz5v3p": "split1_eu",
    "north-america-dxx4sk6tc3": "split1_na",
    "europe-63apu41301": "split2_eu",
    "north-america-j9hk5zna34": "split2_na",
}
TOKEN_ENV_NAMES = ("BALLCHASING_TOKEN", "BALLCHASING_API_TOKEN")


class BallchasingError(RuntimeError):
    """Raised when acquisition cannot safely continue."""


def load_ballchasing_token(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | Path | None = ".env",
) -> str | None:
    """Load a token from the process environment, then an ignored dotenv file."""

    variables = os.environ if env is None else env
    for name in TOKEN_ENV_NAMES:
        value = variables.get(name)
        if value and value.strip():
            return value.strip()
    if dotenv_path is None:
        return None
    path = Path(dotenv_path)
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in TOKEN_ENV_NAMES:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            return value
    return None


@dataclass(frozen=True)
class ReplayGroup:
    """One leaf replay group and its full hierarchy."""

    group_id: str
    path_ids: tuple[str, ...]
    path_names: tuple[str, ...]
    root_label: str

    @property
    def group_path(self) -> str:
        return "/".join(self.path_names)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_json(response: Any) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise BallchasingError("Ballchasing returned a non-object JSON response.")
    return payload


class BallchasingClient:
    """Small API client with bounded retries and injectable I/O for tests."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = BALLCHASING_API_BASE,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not token.strip():
            raise ValueError("A non-empty Ballchasing API token is required.")
        if session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "RLCS acquisition requires requests; install footballq[rlcs]."
                ) from exc
            session = requests.Session()
        self.token = token.strip()
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.sleep = sleep
        self.timeout_seconds = float(timeout_seconds)

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        return f"{self.base_url}/{path_or_url.lstrip('/')}"

    def get(
        self,
        path_or_url: str,
        *,
        params: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> Any:
        delays = (2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
        for attempt in range(len(delays) + 1):
            response = self.session.get(
                self._url(path_or_url),
                headers={"Authorization": self.token},
                params=None if path_or_url.startswith("http") else dict(params or {}),
                timeout=self.timeout_seconds,
                stream=stream,
            )
            status = int(response.status_code)
            if 200 <= status < 300:
                return response
            if status == 429 or 500 <= status < 600:
                if attempt >= len(delays):
                    break
                retry_after = response.headers.get("Retry-After") if status == 429 else None
                try:
                    delay = float(retry_after) if retry_after is not None else delays[attempt]
                except (TypeError, ValueError):
                    delay = delays[attempt]
                self.sleep(max(delay, 0.0))
                continue
            try:
                response.raise_for_status()
            except Exception as exc:
                raise BallchasingError(
                    f"Ballchasing request failed with HTTP {status}: {self._url(path_or_url)}"
                ) from exc
            raise BallchasingError(f"Unexpected Ballchasing HTTP status {status}.")
        raise BallchasingError(
            f"Ballchasing request exhausted retries: {self._url(path_or_url)}"
        )

    def ping(self) -> dict[str, Any]:
        """Validate the configured token before any inventory mutation."""

        return _response_json(self.get("/"))

    def _pages(
        self,
        path: str,
        *,
        params: Mapping[str, Any],
    ) -> Iterator[dict[str, Any]]:
        next_url: str | None = path
        next_params: Mapping[str, Any] | None = params
        seen_urls: set[str] = set()
        while next_url:
            response = self.get(next_url, params=next_params)
            payload = _response_json(response)
            yield payload
            candidate = payload.get("next")
            next_url = str(candidate) if candidate else None
            next_params = None
            if next_url:
                if next_url in seen_urls:
                    raise BallchasingError(f"Pagination loop detected at {next_url!r}.")
                seen_urls.add(next_url)

    def child_groups(self, group_id: str, *, page_size: int = 200) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for page in self._pages(
            "/groups",
            params={"group": group_id, "count": int(page_size), "sort-by": "name"},
        ):
            rows.extend(dict(row) for row in page.get("list", []))
        return rows

    def group(self, group_id: str) -> dict[str, Any]:
        return _response_json(self.get(f"/groups/{group_id}"))

    def leaf_groups(
        self,
        root_group_ids: Iterable[str],
        *,
        root_labels: Mapping[str, str] | None = None,
        page_size: int = 200,
    ) -> list[ReplayGroup]:
        """Breadth-first traversal; replay listing is performed only on leaves."""

        labels = {**DEFAULT_ROOT_LABELS, **dict(root_labels or {})}
        leaves: list[ReplayGroup] = []
        seen: set[str] = set()
        queue: deque[ReplayGroup] = deque()
        for root_id in root_group_ids:
            root = self.group(str(root_id))
            root_name = str(root.get("name") or labels.get(str(root_id)) or root_id)
            queue.append(
                ReplayGroup(
                    group_id=str(root_id),
                    path_ids=(str(root_id),),
                    path_names=(root_name,),
                    root_label=labels.get(str(root_id), str(root_id)),
                )
            )
        while queue:
            current = queue.popleft()
            if current.group_id in seen:
                continue
            seen.add(current.group_id)
            children = self.child_groups(current.group_id, page_size=page_size)
            if not children:
                leaves.append(current)
                continue
            for child in sorted(children, key=lambda row: (str(row.get("name")), str(row["id"]))):
                child_id = str(child["id"])
                queue.append(
                    ReplayGroup(
                        group_id=child_id,
                        path_ids=(*current.path_ids, child_id),
                        path_names=(*current.path_names, str(child.get("name") or child_id)),
                        root_label=current.root_label,
                    )
                )
        return leaves

    def replays(self, group_id: str, *, page_size: int = 200) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        params = {
            "group": group_id,
            "count": int(page_size),
            "sort-by": "replay-date",
            "sort-dir": "asc",
        }
        for page in self._pages("/replays", params=params):
            rows.extend(dict(row) for row in page.get("list", []))
        return rows


def _players_from_team(team: Any) -> list[dict[str, Any]]:
    if not isinstance(team, dict):
        return []
    players = team.get("players", [])
    return [dict(player) for player in players if isinstance(player, dict)]


def _player_identity(player: dict[str, Any]) -> dict[str, str | None]:
    identifier = player.get("id")
    platform: str | None = None
    platform_id: str | None = None
    if isinstance(identifier, dict):
        platform = str(identifier.get("platform") or "") or None
        platform_id = str(identifier.get("id") or "") or None
    elif identifier is not None:
        raw = str(identifier)
        if ":" in raw:
            platform, platform_id = raw.split(":", 1)
        else:
            platform_id = raw
    return {
        "name": str(player.get("name") or "") or None,
        "platform": platform,
        "platform_id": platform_id,
    }


def group_taxonomy(group: ReplayGroup) -> dict[str, Any]:
    """Derive frozen split fields from the official group hierarchy."""

    root = group.root_label.casefold()
    split_number = 1 if "split1" in root else 2 if "split2" in root else None
    region = "EU" if root.endswith("eu") else "NA" if root.endswith("na") else None
    names = list(group.path_names)
    regional: int | None = None
    stage: str | None = None
    for name in names[1:]:
        lowered = name.casefold()
        match = re.search(r"(?:regional|open)\s*#?\s*(\d+)", lowered)
        if match:
            regional = int(match.group(1))
        if any(word in lowered for word in ("qualifier", "swiss", "group", "playoff", "final")):
            stage = name
    return {
        "region": region,
        "split_number": split_number,
        "regional_number": regional,
        "stage": stage,
        "series_id": group.group_id,
    }


def inventory_record(replay: dict[str, Any], group: ReplayGroup) -> dict[str, Any]:
    """Flatten one API replay into the immutable inventory schema."""

    taxonomy = group_taxonomy(group)
    blue = replay.get("blue", {})
    orange = replay.get("orange", {})
    players = [
        _player_identity(player)
        for player in (*_players_from_team(blue), *_players_from_team(orange))
    ]
    title = str(replay.get("title") or "")
    game_match = re.search(r"\bgame\s*#?\s*(\d+)\b", title, re.IGNORECASE)
    return {
        "replay_id": str(replay["id"]),
        "rocket_league_game_id": str(replay.get("rocket_league_id") or "") or None,
        "root_group_id": group.path_ids[0],
        "leaf_group_id": group.group_id,
        "group_path": group.group_path,
        **taxonomy,
        "game_number": int(game_match.group(1)) if game_match else None,
        "event_time_utc": replay.get("date") or replay.get("created"),
        "map_name": replay.get("map_name") or replay.get("map"),
        "duration_seconds": replay.get("duration"),
        "blue_score": blue.get("goals") if isinstance(blue, dict) else None,
        "orange_score": orange.get("goals") if isinstance(orange, dict) else None,
        "title": title,
        "players_json": json.dumps(players, sort_keys=True, separators=(",", ":")),
        "download_status": "pending",
        "downloaded_at_utc": None,
        "file_size_bytes": None,
        "file_sha256": None,
    }


def build_inventory(
    client: BallchasingClient,
    root_group_ids: Iterable[str],
    *,
    root_labels: Mapping[str, str] | None = None,
    page_size: int = 200,
) -> list[dict[str, Any]]:
    """Inventory all leaf groups and globally deduplicate replay IDs."""

    by_id: dict[str, dict[str, Any]] = {}
    for leaf in client.leaf_groups(
        root_group_ids,
        root_labels=root_labels,
        page_size=page_size,
    ):
        for replay in client.replays(leaf.group_id, page_size=page_size):
            row = inventory_record(replay, leaf)
            replay_id = str(row["replay_id"])
            if replay_id in by_id:
                old_paths = set(str(by_id[replay_id]["group_path"]).split(" | "))
                old_paths.add(str(row["group_path"]))
                by_id[replay_id]["group_path"] = " | ".join(sorted(old_paths))
            else:
                by_id[replay_id] = row
    return sorted(
        by_id.values(),
        key=lambda row: (str(row.get("event_time_utc") or ""), str(row["replay_id"])),
    )


def write_inventory_parquet(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write inventory atomically before replay bytes are requested."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Inventory writing requires pyarrow.") from exc
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    metadata = dict(table.schema.metadata or {})
    metadata[b"footballq_schema"] = b"rlcs_replay_inventory_v1"
    table = table.replace_schema_metadata(metadata)
    temporary = path_obj.with_suffix(path_obj.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path_obj)
    return path_obj


def read_inventory_parquet(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Inventory reading requires pyarrow.") from exc
    return pq.read_table(path).to_pylist()


class DownloadLimiter:
    """Enforce both per-second and rolling hourly download limits."""

    def __init__(
        self,
        *,
        requests_per_second: float,
        hourly_cap: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.minimum_interval = 1.0 / float(requests_per_second)
        self.hourly_cap = int(hourly_cap)
        self.clock = clock
        self.sleep = sleep
        self.last_request: float | None = None
        self.request_times: deque[float] = deque()

    def seed_recent_downloads(
        self,
        timestamps: Iterable[str | None],
        *,
        now_utc: datetime | None = None,
    ) -> None:
        """Restore the rolling hourly window from persisted inventory timestamps."""

        wall_now = now_utc or datetime.now(UTC)
        monotonic_now = self.clock()
        restored: list[float] = []
        for raw in timestamps:
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            age = max((wall_now - parsed.astimezone(UTC)).total_seconds(), 0.0)
            if age < 3600.0:
                restored.append(monotonic_now - age)
        self.request_times = deque(sorted(restored))
        self.last_request = self.request_times[-1] if self.request_times else None

    def wait(self) -> None:
        now = self.clock()
        while self.request_times and now - self.request_times[0] >= 3600.0:
            self.request_times.popleft()
        delays = [0.0]
        if self.last_request is not None:
            delays.append(self.minimum_interval - (now - self.last_request))
        if len(self.request_times) >= self.hourly_cap:
            delays.append(3600.0 - (now - self.request_times[0]))
        delay = max(delays)
        if delay > 0:
            self.sleep(delay)
            now = self.clock()
            while self.request_times and now - self.request_times[0] >= 3600.0:
                self.request_times.popleft()
        self.last_request = now
        self.request_times.append(now)


def download_replays(
    client: BallchasingClient,
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    requests_per_second: float = 1.0,
    hourly_cap: int = 200,
    resume: bool = True,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Download replay files with content hashes and resumable checkpoints."""

    output = Path(output_dir)
    replay_dir = output / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    limiter = DownloadLimiter(
        requests_per_second=requests_per_second,
        hourly_cap=hourly_cap,
        sleep=client.sleep,
    )
    limiter.seed_recent_downloads(row.get("downloaded_at_utc") for row in records)
    for row in records:
        replay_id = str(row["replay_id"])
        destination = replay_dir / f"{replay_id}.replay"
        if resume and destination.exists():
            digest = sha256_file(destination)
            expected = row.get("file_sha256")
            if expected and str(expected) != digest:
                raise BallchasingError(
                    f"Existing replay hash mismatch for {replay_id}; refusing to overwrite."
                )
            row.update(
                {
                    "download_status": "complete",
                    "downloaded_at_utc": row.get("downloaded_at_utc")
                    or datetime.now(UTC).replace(microsecond=0).isoformat(),
                    "file_size_bytes": destination.stat().st_size,
                    "file_sha256": digest,
                }
            )
            continue
        limiter.wait()
        response = client.get(f"/replays/{replay_id}/file", stream=True)
        temporary = destination.with_suffix(".replay.part")
        digest = hashlib.sha256()
        size = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        temporary.replace(destination)
        row.update(
            {
                "download_status": "complete",
                "downloaded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "file_size_bytes": size,
                "file_sha256": digest.hexdigest(),
            }
        )
        if checkpoint is not None:
            checkpoint(records)
    if checkpoint is not None:
        checkpoint(records)
    return records


def acquire_rlcs(
    *,
    token: str,
    root_group_ids: list[str],
    output_dir: str | Path,
    page_size: int = 200,
    requests_per_second: float = 1.0,
    hourly_cap: int = 200,
    resume: bool = True,
    download_limit: int | None = None,
    client: BallchasingClient | None = None,
) -> Path:
    """Inventory and download the prespecified RLCS corpus."""

    output = Path(output_dir)
    inventory_path = output / "replay_inventory.parquet"
    client = client or BallchasingClient(token)
    client.ping()
    if resume and inventory_path.exists():
        records = read_inventory_parquet(inventory_path)
    else:
        records = build_inventory(client, root_group_ids, page_size=page_size)
        write_inventory_parquet(records, inventory_path)
    selected = sorted(
        records,
        key=lambda row: (
            int(row.get("split_number") or 99),
            str(row.get("region") or ""),
            str(row.get("event_time_utc") or ""),
            str(row["replay_id"]),
        ),
    )
    if download_limit is not None:
        if int(download_limit) < 1:
            raise ValueError("download_limit must be positive when provided.")
        selected = selected[: int(download_limit)]
    download_replays(
        client,
        selected,
        output,
        requests_per_second=requests_per_second,
        hourly_cap=hourly_cap,
        resume=resume,
        checkpoint=lambda _rows: write_inventory_parquet(records, inventory_path),
    )
    return inventory_path
