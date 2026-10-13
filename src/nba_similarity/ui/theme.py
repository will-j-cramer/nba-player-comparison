"""The viewer's light/dark theme, for figure colors."""

from __future__ import annotations

import streamlit as st


def theme_type() -> str:
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", None) or "light"
