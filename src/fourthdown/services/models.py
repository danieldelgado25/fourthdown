"""The models service: the fourth-down advisor, priced in win probability.

Tools:
    ``advisor`` -- parses a fourth-down situation out of a question and ranks go, field
    goal, and punt by the win probability each leaves.

Endpoints, beyond `common.tool_service`'s ``/health`` and ``/tools/advisor``:
    ``POST /advise`` -- a structured situation in, the ranked options out.

Dependencies: the ``models`` extra (torch, scikit-learn). Reads ``FOURTHDOWN_DATA_DIR``
for the artifacts `fourthdown train` writes. Needs no LLM, no warehouse, and no Postgres
at serving time. Default port 8003.

Run: ``python -m fourthdown.services.models`` or ``fourthdown serve models``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from flask import Flask, Response, jsonify, request

from fourthdown.agent.parse import parse_state
from fourthdown.agent.tools import Evidence, Table
from fourthdown.config import Paths, data_paths
from fourthdown.models import fourth_down
from fourthdown.models.fourth_down import FourthDownAdvisor, Recommendation
from fourthdown.models.train import FOURTH_DOWN_ARTIFACT, WP_ARTIFACT
from fourthdown.models.winprob import GameState, WinProbabilityModel
from fourthdown.services.common import (
    Capabilities,
    Reloading,
    configure_logging,
    serve,
    tool_service,
)

NAME = "models"
PORT = 8003


class AdvisorTool:
    """Fourth-down decisions, priced in win probability.

    Args:
        advisor: The loaded advisor (win-probability net plus its component models).

    `run(question)` returns the ranked options as a table, with the situation, best
    option, and edge in ``detail``; failed when the question lacks distance or field
    position, which are never guessed.
    """

    name = "advisor"
    description = (
        "Whether to go for it, kick a field goal, or punt on fourth down, given a "
        "situation: yard line, yards to go, time, and score. Use whenever the question "
        "asks what a team should do on fourth down."
    )

    def __init__(self, advisor: FourthDownAdvisor) -> None:
        self._advisor = advisor

    def run(self, question: str) -> Evidence:
        """Parse the situation and rank the options; see the class docstring."""
        state = parse_state(question)
        if state is None:
            return Evidence(
                answer=(
                    "I need the situation to advise: yard line, yards to go, time left, "
                    "and the score margin."
                ),
                failed=True,
            )
        recommendation = self._advisor.recommend(state)
        return Evidence(
            answer=recommendation.render(),
            table=Table(
                columns=("option", "win probability", "success", "detail"),
                rows=[
                    [
                        option.name,
                        round(option.win_probability, 4),
                        None
                        if option.success_probability is None
                        else round(option.success_probability, 4),
                        option.detail,
                    ]
                    for option in recommendation.options
                ],
            ),
            detail={
                "situation": recommendation.render().splitlines()[0],
                "best": recommendation.best.name,
                "edge": f"{recommendation.edge:.4f}",
            },
        )


def state_from(payload: dict[str, Any]) -> GameState:
    """Validate a ``/advise`` body into a fourth-down `GameState`.

    Args:
        payload: JSON with required ``yardline_100`` (1-99, yards from the opponent's
            goal) and ``ydstogo`` (>= 0), and optional ``game_seconds_remaining``
            (default 900), ``score_differential`` (0), ``posteam_is_home`` (true),
            ``posteam_spread`` (0), and ``goal_to_go`` (inferred).

    Returns:
        The state, always with ``down=4``.

    Raises:
        KeyError: A required field is missing.
        ValueError, TypeError: A field is out of range or not a number.
    """
    yardline = float(payload["yardline_100"])
    togo = float(payload["ydstogo"])
    if not 1.0 <= yardline <= 99.0:
        raise ValueError("yardline_100 must be between 1 and 99")
    if togo < 0.0:
        raise ValueError("ydstogo must not be negative")
    return GameState(
        yardline_100=yardline,
        down=4,
        ydstogo=togo,
        game_seconds_remaining=float(payload.get("game_seconds_remaining", 900.0)),
        score_differential=float(payload.get("score_differential", 0.0)),
        posteam_is_home=bool(payload.get("posteam_is_home", True)),
        posteam_spread=float(payload.get("posteam_spread", 0.0)),
        goal_to_go=bool(payload.get("goal_to_go", yardline <= togo)),
    )


def recommendation_json(recommendation: Recommendation) -> dict[str, Any]:
    """Serialise a `Recommendation` into the ``/advise`` reply.

    Returns:
        ``situation`` (one line), ``best`` (option name), ``edge`` (win-probability gap
        to the runner-up), and ``options`` -- each with ``name``, ``win_probability``,
        ``success_probability`` (null for punts), and ``detail`` -- best first.
    """
    return {
        "situation": recommendation.render().splitlines()[0],
        "best": recommendation.best.name,
        "edge": recommendation.edge,
        "options": [
            {
                "name": option.name,
                "win_probability": option.win_probability,
                "success_probability": option.success_probability,
                "detail": option.detail,
            }
            for option in recommendation.options
        ],
    }


@dataclass
class ModelCapabilities(Capabilities):
    """`Capabilities` plus the loaded advisor, for the ``/advise`` endpoint."""

    advisor: FourthDownAdvisor | None = None


def load_advisor(paths: Paths) -> FourthDownAdvisor | None:
    """Load the trained advisor from ``paths.models``, or ``None`` if not trained yet."""
    weights = paths.models / WP_ARTIFACT
    components = paths.models / FOURTH_DOWN_ARTIFACT
    if not weights.exists() or not components.exists():
        return None
    return fourth_down.load_components(components, WinProbabilityModel.load(weights))


def opener(paths: Paths | None = None) -> Callable[[ExitStack], ModelCapabilities]:
    """Build the function `Reloading` calls to load this service's models.

    Args:
        paths: Data layout; ``FOURTHDOWN_DATA_DIR`` or the repo default when ``None``.

    Returns:
        A callable reporting the ``models`` component; ``advisor`` loads once
        `fourthdown train` has written its artifacts.
    """
    resolved = paths or data_paths()

    def open_(stack: ExitStack) -> ModelCapabilities:
        state = ModelCapabilities()
        state.advisor = load_advisor(resolved)
        state.components["models"] = state.advisor is not None
        if state.advisor is None:
            state.skipped["models"] = f"no trained artifacts in {resolved.models}"
        else:
            state.tools.append(AdvisorTool(state.advisor))
        return state

    return open_


def create_app(resources: Reloading[ModelCapabilities]) -> Flask:
    """The models service app.

    Args:
        resources: The service's reloading resources (see `opener`).

    Returns:
        `common.tool_service` plus ``POST /advise``: body as in `state_from`, reply as in
        `recommendation_json`; 400 on an invalid situation, 503 before training.
    """
    app = tool_service(NAME, resources, errors=())

    @app.post("/advise")
    def advise() -> tuple[Response, int] | Response:
        advisor = resources.current().advisor
        if advisor is None:
            return jsonify({"error": "no trained models; run `fourthdown train`"}), 503
        try:
            state = state_from(request.get_json(silent=True) or {})
        except KeyError as error:
            return jsonify({"error": f"{error.args[0]} is required"}), 400
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(recommendation_json(advisor.recommend(state)))

    return app


def main(host: str | None = None, port: int | None = None) -> None:
    """Load the models and serve until interrupted.

    Args:
        host: Bind address (see `common.serve`).
        port: Port (see `common.serve`); 8003 by default.
    """
    configure_logging()
    resources = Reloading(opener())
    try:
        serve(create_app(resources), host=host, port=port, default_port=PORT)
    finally:
        resources.close()


if __name__ == "__main__":
    main()
