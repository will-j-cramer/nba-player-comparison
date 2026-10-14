"""Plain-text pieces shared by the career CLI and app (no streamlit)."""

from __future__ import annotations

from nba_similarity.careers.engine import CareerComp, CareerTrait
from nba_similarity.careers.outcomes import Outcome
from nba_similarity.similarity.explain import format_value

OUTCOME_MARKS = {"qualified": "●", "limited": "◐", "out": "○"}
OUTCOME_LEGEND = "● full season  ◐ played, under the minimum  ○ not in the NBA  · not played yet"


def format_ages(ages: tuple[int, ...]) -> str:
    """(20, 21) -> "20-21"; (21,) -> "21"."""
    return f"{ages[0]}-{ages[-1]}" if len(ages) > 1 else str(ages[0])


def format_trait(t: CareerTrait) -> str:
    """ "Block % 9.2 vs 7.1 (ages 20-21)": query first, values at the latest shared age."""
    return (
        f"{t.label} {format_value(t.query_raw)} vs {format_value(t.comp_raw)} "
        f"(age{'s' if len(t.ages) > 1 else ''} {format_ages(t.ages)})"
    )


def format_outcome(outcome: Outcome, horizon: int, first_age: int) -> str:
    """ "22 ● 23 ● 24 ◐ 25 ○ 26 ·": one mark per future age."""
    marks = [f"{s.age} {OUTCOME_MARKS[s.outcome]}" for s in outcome.seasons]
    marks += [f"{first_age + i} ·" for i in range(len(outcome.seasons), horizon)]
    return "  ".join(marks)


def comp_summary(comp: CareerComp) -> str:
    """The comp's alike and differ stats, one line each."""
    alike = "; ".join(format_trait(t) for t in comp.similar_traits) or "-"
    differ = "; ".join(format_trait(t) for t in comp.differences)
    return f"alike: {alike}" + (f"\ndiffer: {differ}" if differ else "")
