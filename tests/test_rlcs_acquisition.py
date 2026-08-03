from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from footballq.data.rlcs_ballchasing import (
    BallchasingClient,
    DownloadLimiter,
    ReplayGroup,
    build_inventory,
    download_replays,
    inventory_record,
    load_ballchasing_token,
)


class Response:
    def __init__(self, status: int, payload: dict[str, Any] | None = None, body: bytes = b""):
        self.status_code = status
        self._payload = payload or {}
        self.headers: dict[str, str] = {}
        self.body = body

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.body


class Session:
    def __init__(self, responses: list[Response]):
        self.responses = responses
        self.calls = []

    def get(self, url: str, **kwargs: Any) -> Response:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_token_loader_prefers_process_environment(tmp_path: Path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("BALLCHASING_TOKEN=file-token\n", encoding="utf-8")
    assert (
        load_ballchasing_token(
            {"BALLCHASING_TOKEN": " process-token "}, dotenv_path=dotenv
        )
        == "process-token"
    )


def test_token_loader_reads_ignored_dotenv_without_emitting_value(
    tmp_path: Path, capsys
):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# local secret\nBALLCHASING_API_TOKEN='file-token'\n", encoding="utf-8"
    )
    assert load_ballchasing_token({}, dotenv_path=dotenv) == "file-token"
    captured = capsys.readouterr()
    assert "file-token" not in captured.out
    assert "file-token" not in captured.err


def test_client_honors_retry_after_and_authorization():
    limited = Response(429)
    limited.headers["Retry-After"] = "3"
    session = Session([limited, Response(200, {"ok": True})])
    sleeps: list[float] = []
    client = BallchasingClient("secret", session=session, sleep=sleeps.append)
    assert client.ping() == {"ok": True}
    assert sleeps == [3.0]
    assert session.calls[0][1]["headers"] == {"Authorization": "secret"}


def test_inventory_deduplicates_replay_ids_across_leaf_groups():
    class Client:
        def leaf_groups(self, *_args: Any, **_kwargs: Any):
            return [
                ReplayGroup("leaf-a", ("root", "leaf-a"), ("Root", "A"), "split1_eu"),
                ReplayGroup("leaf-b", ("root", "leaf-b"), ("Root", "B"), "split1_eu"),
            ]

        def replays(self, group_id: str, **_kwargs: Any):
            return [
                {
                    "id": "replay-1",
                    "date": "2025-01-01T00:00:00Z",
                    "title": "Game 1",
                    "blue": {"players": []},
                    "orange": {"players": []},
                }
            ]

    rows = build_inventory(Client(), ["root"])
    assert len(rows) == 1
    assert rows[0]["group_path"] == "Root/A | Root/B"


def test_inventory_flattens_platform_identifiers():
    group = ReplayGroup(
        "series", ("root", "series"), ("Europe", "Regional 2"), "split2_eu"
    )
    row = inventory_record(
        {
            "id": "r1",
            "title": "Final Game #3",
            "blue": {
                "goals": 2,
                "players": [{"name": "A", "id": {"platform": "steam", "id": "1"}}],
            },
            "orange": {"goals": 1, "players": []},
        },
        group,
    )
    assert row["region"] == "EU"
    assert row["split_number"] == 2
    assert row["regional_number"] == 2
    assert row["game_number"] == 3
    assert '"platform_id":"1"' in row["players_json"]


def test_download_resume_hashes_existing_file_without_request(tmp_path: Path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    (replay_dir / "r1.replay").write_bytes(b"native replay")
    client = BallchasingClient("token", session=Session([]), sleep=lambda _value: None)
    records = [{"replay_id": "r1", "file_sha256": None}]
    download_replays(client, records, tmp_path, resume=True)
    assert records[0]["download_status"] == "complete"
    assert records[0]["file_size_bytes"] == len(b"native replay")


def test_download_limiter_enforces_interval_with_injected_clock():
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(value: float) -> None:
        now[0] += value

    limiter = DownloadLimiter(requests_per_second=1.0, hourly_cap=200, clock=clock, sleep=sleep)
    limiter.wait()
    limiter.wait()
    assert now[0] == 1.0


def test_download_limiter_restores_rolling_hour_after_restart():
    now = [1000.0]

    def clock() -> float:
        return now[0]

    def sleep(value: float) -> None:
        now[0] += value

    wall_now = datetime(2026, 8, 2, tzinfo=UTC)
    limiter = DownloadLimiter(
        requests_per_second=1.0,
        hourly_cap=1,
        clock=clock,
        sleep=sleep,
    )
    limiter.seed_recent_downloads(
        [
            (wall_now - timedelta(seconds=10)).isoformat(),
            (wall_now - timedelta(seconds=3700)).isoformat(),
        ],
        now_utc=wall_now,
    )
    limiter.wait()
    assert now[0] == 4590.0
