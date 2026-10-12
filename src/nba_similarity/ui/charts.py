"""Plotly figure builders for the app (pure: data in, figure out; no streamlit)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from nba_similarity.ui.view import RadarData

# Categorical slots 1 and 2 of the validated reference palette (light / dark steps); both
# pairs pass the CVD, normal-vision and contrast checks against their surfaces.
QUERY_COLOR = {"light": "#2a78d6", "dark": "#3987e5"}
MATCH_COLOR = {"light": "#eb6834", "dark": "#d95926"}
TEXT_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}
GRID = {"light": "rgba(82, 81, 78, 0.18)", "dark": "rgba(195, 194, 183, 0.18)"}


def _closed(values: np.ndarray) -> list[float | None]:
    """Close the polygon; NaN becomes None so Plotly leaves a gap instead of plotting 0."""
    out = [None if np.isnan(v) else float(v) for v in values]
    return out + out[:1]


def _hover(labels: tuple[str, ...], raw: np.ndarray, pct: np.ndarray) -> list[str]:
    text = [
        f"{label}: no data" if np.isnan(p) else f"{label}: {r:.3g} ({p:.0f}th pct of season)"
        for label, r, p in zip(labels, raw, pct, strict=True)
    ]
    return text + text[:1]


def radar_figure(
    data: RadarData, query_label: str, match_label: str, theme: str = "light"
) -> go.Figure:
    """Two overlaid profiles on a 0-100 percentile scale; missing stats are gaps, not zeros."""
    mode = "dark" if theme == "dark" else "light"
    theta = list(data.labels) + [data.labels[0]]
    fig = go.Figure()
    for name, pct, raw, color in (
        (query_label, data.query_pct, data.query_raw, QUERY_COLOR[mode]),
        (match_label, data.match_pct, data.match_raw, MATCH_COLOR[mode]),
    ):
        fig.add_trace(
            go.Scatterpolar(
                r=_closed(pct),
                theta=theta,
                name=name,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 8, "color": color},
                # a filled area would bridge missing stats and imply values that don't exist
                fill="none" if np.isnan(pct).any() else "toself",
                fillcolor=_with_alpha(color, 0.12),
                connectgaps=False,
                hovertext=_hover(data.labels, raw, pct),
                hoverinfo="text+name",
            )
        )
    fig.update_layout(
        polar={
            "radialaxis": {
                "range": [0, 100],
                "tickvals": [25, 50, 75, 100],
                "ticksuffix": "",
                "gridcolor": GRID[mode],
                "linecolor": GRID[mode],
                "tickfont": {"size": 10, "color": TEXT_SECONDARY[mode]},
                "angle": 90,
            },
            "angularaxis": {
                "gridcolor": GRID[mode],
                "linecolor": GRID[mode],
                "tickfont": {"size": 11, "color": TEXT_SECONDARY[mode]},
            },
            "bgcolor": "rgba(0,0,0,0)",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "x": 0.5, "xanchor": "center"},
        margin={"l": 50, "r": 50, "t": 30, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"namelength": -1},
    )
    return fig


def _with_alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"
