"""Plotly figure builders for the app (pure: data in, figure out; no streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------- map

# Clusters: the first three categorical slots are the only ones that stay distinguishable
# as *all* pairs (a scatter puts every cluster next to every other); validated with
# validate_palette.js --pairs all in both modes (worst CVD dE 9.2 light / 9.4 dark). More
# clusters reuse the colors with a second marker shape, and every cluster carries a direct
# number label at its centroid, so identity never depends on color alone.
CLUSTER_COLORS = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a"),
    "dark": ("#3987e5", "#d95926", "#199e70"),
}
CLUSTER_SYMBOLS = ("circle", "square", "diamond")
MAX_CLUSTERS = len(CLUSTER_SYMBOLS) * 3

# Stat percentiles: single-hue blue ramp (steps 200 -> 700 on light, 600 -> 100 on dark), so
# low values recede toward the surface but every point stays visible.
_BLUE = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SEQUENTIAL = {
    "light": [[i / 5, c] for i, c in enumerate(_BLUE)],
    "dark": [[i / 5, c] for i, c in enumerate(("#184f95", "#1c5cab", "#2a78d6", "#5598e7",
                                               "#9ec5f4", "#cde2fb"))],
}  # fmt: skip
NO_DATA = {"light": "#b4b2ab", "dark": "#5f5e59"}
INK = {"light": "#1a1a19", "dark": "#f0efec"}  # highlight markers: neutral, never a data color
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}
DIMMED_OPACITY = 0.45


def cluster_style(cluster: int, theme: str = "light") -> tuple[str, str]:
    """(color, marker symbol) of a 0-based cluster."""
    mode = "dark" if theme == "dark" else "light"
    return CLUSTER_COLORS[mode][cluster % 3], CLUSTER_SYMBOLS[(cluster // 3) % 3]


def map_figure(
    frame: pd.DataFrame,
    *,
    color_by: str = "stat",
    color_label: str = "",
    theme: str = "light",
) -> go.Figure:
    """The player-season map: every season as a dimmed point, the query and comps on top.

    color_by "stat" colors by frame.color_pct (gray = no data), "cluster" by frame.cluster.
    """
    mode = "dark" if theme == "dark" else "light"
    fig = go.Figure()
    base = {"mode": "markers", "hovertemplate": "%{hovertext}<extra></extra>"}
    if color_by == "cluster":
        for c in sorted(frame["cluster"].unique()):
            part = frame[frame["cluster"] == c]
            color, symbol = cluster_style(int(c), mode)
            fig.add_trace(
                go.Scattergl(
                    x=part["x"], y=part["y"], hovertext=part["hover"],
                    name=f"{c + 1}: {part['cluster_name'].iloc[0]}",
                    marker={"size": 5, "color": color, "symbol": symbol,
                            "opacity": DIMMED_OPACITY},
                    **base,
                )
            )  # fmt: skip
        centroids = frame.groupby("cluster")[["x", "y"]].median()
        for c, (cx, cy) in centroids.iterrows():
            fig.add_annotation(
                x=cx, y=cy, text=f"<b>{c + 1}</b>", showarrow=False,
                font={"size": 15, "color": INK[mode]},
                bgcolor=_with_alpha(SURFACE[mode], 0.75), borderpad=2,
            )  # fmt: skip
    else:
        has = frame["color_pct"].notna()
        if (~has).any():
            part = frame[~has]
            fig.add_trace(
                go.Scattergl(
                    x=part["x"], y=part["y"], hovertext=part["hover"], name="No data",
                    marker={"size": 5, "color": NO_DATA[mode], "opacity": DIMMED_OPACITY},
                    **base,
                )
            )  # fmt: skip
        part = frame[has]
        fig.add_trace(
            go.Scattergl(
                x=part["x"], y=part["y"], hovertext=part["hover"], name=color_label,
                showlegend=False,
                marker={
                    "size": 5, "color": part["color_pct"], "cmin": 0, "cmax": 100,
                    "colorscale": SEQUENTIAL[mode], "opacity": DIMMED_OPACITY + 0.15,
                    "colorbar": {"title": {"text": f"{color_label}<br>pct of season"},
                                 "tickvals": [0, 25, 50, 75, 100], "thickness": 12},
                },
                **base,
            )
        )  # fmt: skip

    comps = frame[frame["role"] == "comp"].sort_values("rank")
    query = frame[frame["role"] == "query"]
    # The query and its comps tend to sit close together, so labels are short (rank numbers;
    # the page lists the names) and each goes to the first side where it overlaps nothing.
    hl = pd.concat([query, comps])
    texts = [f"<b>{lbl}</b>" for lbl in query["label"]] + [str(r) for r in comps["rank"]]
    widths = [len(lbl) for lbl in query["label"]] + [len(t) for t in texts[len(query) :]]
    positions = label_positions(hl["x"].to_numpy(), hl["y"].to_numpy(), widths, _px_per_unit(frame))
    if len(comps):
        fig.add_trace(
            go.Scatter(
                x=comps["x"], y=comps["y"], hovertext=comps["hover"], name="Top comps (rank)",
                mode="markers+text", text=texts[len(query) :],
                textposition=positions[len(query) :],
                textfont={"size": 12, "color": INK[mode]},
                marker={"size": 12, "color": "rgba(0,0,0,0)",
                        "line": {"color": INK[mode], "width": 2}},
                hovertemplate="%{hovertext}<extra>comp</extra>",
            )
        )  # fmt: skip
    if len(query):
        fig.add_trace(
            go.Scatter(
                x=query["x"], y=query["y"], hovertext=query["hover"], name="Selected season",
                mode="markers+text", text=texts[: len(query)],
                textposition=positions[: len(query)], textfont={"size": 13, "color": INK[mode]},
                marker={"size": 17, "color": INK[mode], "symbol": "star",
                        "line": {"color": SURFACE[mode], "width": 2}},
                hovertemplate="%{hovertext}<extra>selected</extra>",
            )
        )  # fmt: skip

    hidden_axis = {"visible": False, "showgrid": False, "zeroline": False}
    fig.update_layout(
        xaxis=hidden_axis,
        yaxis={**hidden_axis, "scaleanchor": "x"},  # UMAP axes have no units; keep 1:1
        legend={"orientation": "h", "yanchor": "top", "y": -0.02, "x": 0, "itemsizing": "constant"},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"namelength": -1, "align": "left"},
        dragmode="pan",
    )
    return fig


# Candidate label sides, in order of preference, with the label box's offset from the point
# in pixels (screen y grows downward).
_SIDES = (
    ("top center", 0.0, -1.0), ("middle right", 1.0, 0.0), ("bottom center", 0.0, 1.0),
    ("middle left", -1.0, 0.0), ("top right", 1.0, -1.0), ("bottom right", 1.0, 1.0),
    ("top left", -1.0, -1.0), ("bottom left", -1.0, 1.0),
)  # fmt: skip
_CHAR_PX, _LINE_PX, _MARKER_PX = 7.5, 15.0, 14.0
_PLOT_PX = 640.0  # approximate size of the map's larger side on screen


def _px_per_unit(frame: pd.DataFrame) -> float:
    span = max(np.ptp(frame["x"]), np.ptp(frame["y"])) if len(frame) else 0.0
    return _PLOT_PX / span if span > 0 else 1.0


def label_positions(
    x: np.ndarray, y: np.ndarray, widths: list[int], px_per_unit: float
) -> list[str]:
    """Greedy label placement: each label takes the first side where its box overlaps no
    marker and no label placed before it (falls back to "top center")."""
    px, py = np.asarray(x) * px_per_unit, -np.asarray(y) * px_per_unit  # screen coordinates
    half = _MARKER_PX / 2
    taken = [(cx - half, cy - half, cx + half, cy + half) for cx, cy in zip(px, py, strict=True)]
    out = []
    for cx, cy, n_chars in zip(px, py, widths, strict=True):
        w, h = n_chars * _CHAR_PX + 4, _LINE_PX
        choice = _SIDES[0][0]
        for side, dx, dy in _SIDES:
            # box centre: beside the marker on that side
            bx = cx + dx * (half + w / 2) if dx else cx
            by = cy + dy * (half + h / 2) if dy else cy
            box = (bx - w / 2, by - h / 2, bx + w / 2, by + h / 2)
            if not any(_overlaps(box, t) for t in taken):
                choice = side
                taken.append(box)
                break
        out.append(choice)
    return out


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
