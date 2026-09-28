"""Order gateway targets that share a model name by where the model runs best.

When a requested name matches more than one target (a local server and a
remote endpoint both serving ``qwen3-30b``, say), the fleet view decides:
a machine with the model loaded and serving comes first, then machines that
answer but don't have it loaded (fastest first, within the fleet's own
ranking), then machines the fleet can't see, and paused machines last.

It only reorders targets that already match the name, never starts anything,
and never waits on the network: it reads the fleet view already cached
(refreshed in the background) and, while there is none, leaves the order as
it was. It is on only while ``fleet_peers`` is configured and
``gateway_placement`` is not turned off, so a single-machine gateway routes
exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

from .. import fleet
from ..config import AppConfig

LOCAL = "@local"
# Rank groups; a lower group routes first.
LOADED, AVAILABLE, UNKNOWN, PAUSED = 0, 1, 2, 3
GROUP_LABELS = {LOADED: "loaded here", AVAILABLE: "not loaded here", UNKNOWN: "no fleet data", PAUSED: "paused here"}


def url_host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").casefold()
    except ValueError:
        return ""


@dataclass(frozen=True)
class Rank:
    group: int
    position: int
    host: str
    reason: str

    def key(self) -> tuple[int, int]:
        return (self.group, self.position)


def host_keys(host: dict[str, Any]) -> set[str]:
    """Every key a fleet host can be recognised by."""
    if host.get("local"):
        return {LOCAL}
    keys = {str(host.get("name") or "").casefold(), url_host(str(host.get("url") or ""))}
    return {k for k in keys if k}


def rank_hosts(overview: dict[str, Any], model: str) -> dict[str, Rank]:
    """Fleet host key -> rank for ``model`` (a profile name or model file stem)."""
    result = fleet.placement(overview, profile=model, model=model)
    ranks: dict[str, Rank] = {}
    for position, candidate in enumerate(result["candidates"]):
        tier = candidate["tier"]
        group = LOADED if tier == 0 else PAUSED if tier == 1 else AVAILABLE
        reason = "; ".join(candidate["reasons"])
        host = next((h for h in overview.get("hosts") or [] if h["name"] == candidate["host"]), {})
        for key in host_keys(host):
            ranks.setdefault(key, Rank(group, position, candidate["host"], reason))
    return ranks


class FleetRanker:
    """Ranks targets by the fleet view; ``None`` from ``ranks`` means leave the order alone."""

    def __init__(
        self,
        config: Callable[[], AppConfig] = AppConfig.load,
        get_fleet: Callable[[list[Any], str], fleet.Fleet] = fleet.current,
    ) -> None:
        self._config = config
        self._get_fleet = get_fleet

    def ranks(self, model: str) -> dict[str, Rank] | None:
        try:
            config = self._config()
            if not config.fleet_peers or not getattr(config, "gateway_placement", True):
                return None
            overview = self._get_fleet(config.fleet_peers, config.fleet_name).overview_nowait()
            return rank_hosts(overview, model) if overview else None
        except Exception:
            return None  # telemetry trouble never fails or reroutes a request


def rank_target(keys: tuple[str, ...], ranks: dict[str, Rank]) -> Rank:
    for key in keys:
        if key.casefold() in ranks:
            return ranks[key.casefold()]
    return Rank(UNKNOWN, 0, "", "no fleet data for this target's machine")
