from __future__ import annotations

import pytest

from nba_similarity.ingestion.records import COMBINED_TEAM, RawRow
from nba_similarity.ingestion.transform import build_season, collapse_traded, is_combined_team


def row(player_id: str, team: str, g: str = "10", mp: str = "200", **stats: str) -> RawRow:
    return {
        "player_id": player_id,
        "name": player_id.title(),
        "age": "25",
        "pos": "SF",
        "team": team,
        "g": g,
        "gs": "5",
        "mp": mp,
        **stats,
    }


@pytest.mark.parametrize(
    ("team", "expected"),
    [("TOT", True), ("2TM", True), ("3TM", True), ("BOS", False), ("", False), (None, False)],
)
def test_is_combined_team(team: str | None, expected: bool) -> None:
    assert is_combined_team(team) is expected


@pytest.mark.parametrize("combined_label", ["TOT", "2TM"])
def test_collapse_keeps_combined_row_and_stints(combined_label: str) -> None:
    rows = [
        row("a", combined_label, g="60"),
        row("a", "BRK", g="40"),
        row("a", "PHO", g="20"),
        row("b", "BOS"),
    ]
    collapsed = collapse_traded(rows)
    season_row, stints = collapsed["a"]
    assert season_row["g"] == "60"
    assert [s["team"] for s in stints] == ["BRK", "PHO"]
    assert collapsed["b"][1] == [collapsed["b"][0]]


def test_collapse_rejects_team_rows_without_combined_row() -> None:
    with pytest.raises(ValueError, match="no combined row"):
        collapse_traded([row("a", "BRK"), row("a", "PHO")])


def test_build_season_merges_pages_and_normalizes_traded_team() -> None:
    per100 = [row("a", "2TM", pts="30.1"), row("a", "BRK"), row("a", "PHO"), row("b", "BOS")]
    advanced = [
        row("a", "2TM", g="47", mp="1672", usg_pct="31.2", ts_pct=".677"),
        row("a", "BRK", g="39", mp="1400"),
        row("a", "PHO", g="8", mp="272"),
        row("b", "BOS", usg_pct=""),
    ]
    by_id = {ps.player_id: ps for ps in build_season(2023, per100, advanced)}

    a = by_id["a"]
    assert (a.team, a.n_teams, a.g, a.mp, a.season) == (COMBINED_TEAM, 2, 47, 1672, 2023)
    assert [(s.team, s.g, s.mp) for s in a.stints] == [("BRK", 39, 1400), ("PHO", 8, 272)]
    assert a.per100 is not None and a.per100["pts"] == 30.1
    assert a.advanced is not None
    assert a.advanced["usg_pct"] == 31.2
    assert a.advanced["ts_pct"] == 0.677

    b = by_id["b"]
    assert (b.team, b.n_teams) == ("BOS", 1)
    assert b.advanced is not None and b.advanced["usg_pct"] is None  # empty cell -> NULL, not 0
    assert b.per100 is not None and b.per100["fg3"] is None  # column absent -> NULL


def test_build_season_keeps_player_missing_from_one_page() -> None:
    result = build_season(2023, [row("a", "BOS")], [row("a", "BOS"), row("c", "LAL")])
    c = next(ps for ps in result if ps.player_id == "c")
    assert c.per100 is None and c.advanced is not None
