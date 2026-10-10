"""Maps Basketball-Reference `data-stat` cell attributes to our column names.

Checked against the live league pages (NBA_{year}_per_poss.html / NBA_{year}_advanced.html)
in October 2026. Cells whose data-stat is not listed here (ranker, efg_pct, awards) are ignored;
columns a season did not track are simply absent from its page and end up NULL.
"""

from __future__ import annotations

# Cell holding the player's name; its `data-append-csv` attribute is the B-Ref player id.
NAME_STAT = "name_display"
PLAYER_ID_ATTR = "data-append-csv"

IDENTITY_STATS: dict[str, str] = {
    NAME_STAT: "name",
    "age": "age",
    "team_name_abbr": "team",
    "pos": "pos",
    "games": "g",
    "games_started": "gs",
    "mp": "mp",
}

PER100_STATS: dict[str, str] = {
    "fg_per_poss": "fg",
    "fga_per_poss": "fga",
    "fg_pct": "fg_pct",
    "fg3_per_poss": "fg3",
    "fg3a_per_poss": "fg3a",
    "fg3_pct": "fg3_pct",
    "fg2_per_poss": "fg2",
    "fg2a_per_poss": "fg2a",
    "fg2_pct": "fg2_pct",
    "ft_per_poss": "ft",
    "fta_per_poss": "fta",
    "ft_pct": "ft_pct",
    "orb_per_poss": "orb",
    "drb_per_poss": "drb",
    "trb_per_poss": "trb",
    "ast_per_poss": "ast",
    "stl_per_poss": "stl",
    "blk_per_poss": "blk",
    "tov_per_poss": "tov",
    "pf_per_poss": "pf",
    "pts_per_poss": "pts",
    "off_rtg": "ortg",
    "def_rtg": "drtg",
}

ADVANCED_STATS: dict[str, str] = {
    "per": "per",
    "ts_pct": "ts_pct",
    "fg3a_per_fga_pct": "fg3a_rate",
    "fta_per_fga_pct": "fta_rate",
    "orb_pct": "orb_pct",
    "drb_pct": "drb_pct",
    "trb_pct": "trb_pct",
    "ast_pct": "ast_pct",
    "stl_pct": "stl_pct",
    "blk_pct": "blk_pct",
    "tov_pct": "tov_pct",
    "usg_pct": "usg_pct",
    "ows": "ows",
    "dws": "dws",
    "ws": "ws",
    "ws_per_48": "ws_per_48",
    "obpm": "obpm",
    "dbpm": "dbpm",
    "bpm": "bpm",
    "vorp": "vorp",
}
