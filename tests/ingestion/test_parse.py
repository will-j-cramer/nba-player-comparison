from __future__ import annotations

import pytest

from nba_similarity.ingestion.columns import ADVANCED_STATS, PER100_STATS
from nba_similarity.ingestion.parse import ParseError, parse_page
from nba_similarity.ingestion.records import ADVANCED_COLUMNS, PER100_COLUMNS
from tests.ingestion.conftest import read_fixture


def test_column_maps_match_schema_columns() -> None:
    assert tuple(PER100_STATS.values()) == PER100_COLUMNS
    assert tuple(ADVANCED_STATS.values()) == ADVANCED_COLUMNS


def test_parses_player_rows_and_drops_league_average() -> None:
    rows = parse_page(read_fixture("NBA_2023_advanced.html.gz"), "advanced")
    assert len(rows) == 679  # 680 table rows minus "League Average"
    assert all(r["player_id"] for r in rows)
    assert not any(r.get("name") == "League Average" for r in rows)


def test_maps_cells_to_our_column_names() -> None:
    rows = parse_page(read_fixture("NBA_2023_per_poss.html.gz"), "per_poss")
    jokic = next(r for r in rows if r["player_id"] == "jokicni01")
    assert jokic["name"] == "Nikola Jokić"
    assert (jokic["team"], jokic["g"], jokic["mp"]) == ("DEN", "69", "2323")
    assert jokic["pts"] == "35.6"


def test_keeps_traded_players_combined_and_team_rows() -> None:
    rows = parse_page(read_fixture("NBA_2023_advanced.html.gz"), "advanced")
    durant = [(r["team"], r["g"]) for r in rows if r["player_id"] == "duranke01"]
    assert durant == [("2TM", "47"), ("BRK", "39"), ("PHO", "8")]


def test_columns_absent_in_early_seasons_are_missing_and_empty_cells_none() -> None:
    rows = parse_page(read_fixture("NBA_1974_advanced.html"), "advanced")
    hayes = next(r for r in rows if r["player_id"] == "hayesel01")
    assert "usg_pct" not in hayes and "fg3a_rate" not in hayes  # not tracked in 1973-74
    assert hayes["tov_pct"] is None and hayes["gs"] is None  # present but empty
    assert hayes["ts_pct"] == ".470"


def test_missing_table_raises() -> None:
    with pytest.raises(ParseError, match="no <table"):
        parse_page("<html><body><p>Rate limited</p></body></html>", "advanced")


def test_wrong_page_type_raises() -> None:
    with pytest.raises(ParseError):
        parse_page(read_fixture("NBA_1974_advanced.html"), "per_poss")
