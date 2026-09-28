"""Unit tests for the phase-04 models on synthetic play frames.

The frames here are generated from a known rule, so the tests can assert on behaviour
(a model that learns the rule, a split that never shares a season, a state transition
that mirrors the field) rather than on numbers from a particular training run.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from fourthdown.models import features, fourth_down, metrics, playcall, winprob
from fourthdown.models.winprob import GameState

SEASONS = (2009, 2010, 2019, 2021)
PER_SEASON = 400


def synthetic_plays(seed: int = 3) -> pl.DataFrame:
    """Plays whose outcome is a noisy but learnable function of score and clock."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in SEASONS:
        for index in range(PER_SEASON):
            score = float(rng.integers(-21, 22))
            clock = float(rng.integers(0, 3600))
            yardline = float(rng.integers(1, 100))
            togo = float(rng.integers(1, 16))
            down = int(rng.integers(1, 5))
            logit = 0.18 * score * (1.0 - clock / 3600.0) + 0.02 * (50.0 - yardline)
            won = int(rng.random() < 1.0 / (1.0 + np.exp(-logit)))
            pass_logit = 0.35 * togo - 1.4 + 0.06 * -score
            is_pass = int(rng.random() < 1.0 / (1.0 + np.exp(-pass_logit)))
            rows.append(
                {
                    "game_id": f"{season}_{index // 10:02d}",
                    "play_id": float(index),
                    "season": season,
                    "week": 1 + index % 17,
                    "season_type": "REG",
                    "posteam": f"T{index % 4}",
                    "defteam": f"T{(index + 1) % 4}",
                    "qtr": 1 + int(clock // 900),
                    "down": down,
                    "ydstogo": togo,
                    "yardline_100": yardline,
                    "goal_to_go": int(yardline <= togo),
                    "game_seconds_remaining": clock,
                    "half_seconds_remaining": min(clock, 1800.0),
                    "score_differential": score,
                    "posteam_timeouts_remaining": 3,
                    "defteam_timeouts_remaining": 3,
                    "posteam_is_home": index % 2,
                    "shotgun": is_pass,
                    "no_huddle": 0,
                    "posteam_spread": 1.5,
                    "posteam_won": won,
                    "is_designed_play": True,
                    "is_neutral_script": down <= 2,
                    "is_pass_call": is_pass,
                    "first_down": int(rng.random() < 0.35),
                    "touchdown": 0,
                    "field_goal_attempt": False,
                    "punt_attempt": False,
                    "epa": float(rng.normal(0.0, 0.5)),
                    "wp": float(rng.random()),
                    "vegas_wp": float(rng.random()),
                    "xpass": float(rng.random()),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def plays() -> pl.DataFrame:
    return synthetic_plays()


@pytest.fixture(scope="module")
def wp_dataset(plays: pl.DataFrame) -> features.Dataset:
    return features.split_by_season(
        plays, features.WP_FEATURES, "posteam_won", train=(2009, 2010), valid=(2019,), test=(2021,)
    )


@pytest.fixture(scope="module")
def trained(wp_dataset: features.Dataset) -> winprob.WinProbabilityModel:
    model, _ = winprob.train(wp_dataset, max_epochs=25, batch_size=256, patience=25)
    return model


def test_splits_share_no_season(wp_dataset: features.Dataset) -> None:
    assert wp_dataset.train.seasons == (2009, 2010)
    assert wp_dataset.valid.seasons == (2019,)
    assert wp_dataset.test.seasons == (2021,)
    assert not set(wp_dataset.train.seasons) & set(wp_dataset.test.seasons)


def test_feature_matrix_rejects_fitted_model_columns(plays: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="model outputs"):
        features.feature_matrix(plays, ["yardline_100", "vegas_wp"])


def test_feature_matrix_reports_missing_columns(plays: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing feature columns"):
        features.feature_matrix(plays, ["not_a_column"])


def test_split_drops_rows_with_missing_features(plays: pl.DataFrame) -> None:
    holed = plays.with_columns(
        pl.when(pl.col("play_id") == 0.0)
        .then(None)
        .otherwise(pl.col("yardline_100"))
        .alias("yardline_100")
    )
    dataset = features.split_by_season(
        holed, features.WP_FEATURES, "posteam_won", train=(2009,), valid=(2019,), test=(2021,)
    )
    assert len(dataset.train) == PER_SEASON - 1


def test_game_state_round_trips_through_features() -> None:
    state = GameState(
        yardline_100=40.0, down=4, ydstogo=2.0, game_seconds_remaining=600.0, score_differential=-3
    )
    row = state.as_features(features.WP_FEATURES)
    assert row.shape == (1, len(features.WP_FEATURES))
    assert row[0, features.WP_FEATURES.index("score_differential")] == pytest.approx(-3.0)


def test_possession_flip_mirrors_the_field_and_the_scoreboard() -> None:
    state = GameState(
        yardline_100=30.0,
        down=4,
        ydstogo=5.0,
        game_seconds_remaining=900.0,
        score_differential=7.0,
        posteam_timeouts_remaining=1,
        defteam_timeouts_remaining=3,
        posteam_is_home=True,
        posteam_spread=-2.5,
    )
    flipped = state.possession_flip(70.0)
    assert flipped.score_differential == -7.0
    assert flipped.posteam_spread == 2.5
    assert flipped.posteam_is_home is False
    assert flipped.posteam_timeouts_remaining == 3
    assert flipped.defteam_timeouts_remaining == 1
    assert flipped.down == 1
    assert flipped.game_seconds_remaining < state.game_seconds_remaining


def test_replace_changes_one_field_only() -> None:
    state = GameState(
        yardline_100=50.0, down=4, ydstogo=1.0, game_seconds_remaining=120.0, score_differential=0
    )
    assert replace(state, down=1).down == 1
    assert replace(state, down=1).yardline_100 == 50.0


def test_win_probability_learns_the_generating_rule(
    trained: winprob.WinProbabilityModel,
) -> None:
    ahead = GameState(
        yardline_100=50.0, down=1, ydstogo=10.0, game_seconds_remaining=120.0, score_differential=14
    )
    behind = replace(ahead, score_differential=-14)
    assert trained.probability(ahead) > 0.8
    assert trained.probability(behind) < 0.2


def test_win_probability_beats_a_constant_guess(
    trained: winprob.WinProbabilityModel, wp_dataset: features.Dataset
) -> None:
    truth = features.target_vector(wp_dataset.test.frame, wp_dataset.target)
    predicted = trained.predict(features.feature_matrix(wp_dataset.test.frame, wp_dataset.features))
    fitted = metrics.score(truth, predicted)
    constant = metrics.score(truth, np.full(truth.shape, float(truth.mean())))
    assert fitted.log_loss < constant.log_loss
    assert fitted.auc > 0.6


def test_checkpoint_round_trip_is_exact(
    trained: winprob.WinProbabilityModel, wp_dataset: features.Dataset, tmp_path
) -> None:
    path = tmp_path / "winprob.pt"
    trained.save(path)
    reloaded = winprob.WinProbabilityModel.load(path)
    matrix = features.feature_matrix(wp_dataset.test.frame.head(64), wp_dataset.features)
    assert reloaded.features == trained.features
    np.testing.assert_allclose(reloaded.predict(matrix), trained.predict(matrix), rtol=1e-6)


def test_score_rewards_the_better_probabilities() -> None:
    truth = np.array([0.0, 0.0, 1.0, 1.0])
    sharp = metrics.score(truth, np.array([0.05, 0.1, 0.9, 0.95]), bins=4)
    blunt = metrics.score(truth, np.array([0.5, 0.5, 0.5, 0.5]), bins=4)
    assert sharp.log_loss < blunt.log_loss
    assert sharp.brier < blunt.brier
    assert sharp.auc == pytest.approx(1.0)


def test_calibration_error_punishes_overconfidence() -> None:
    truth = np.concatenate([np.ones(30), np.zeros(70)])
    honest = metrics.score(truth, np.full(100, 0.3), bins=10)
    overconfident = metrics.score(truth, np.full(100, 0.9), bins=10)
    assert honest.calibration_error == pytest.approx(0.0, abs=1e-9)
    assert overconfident.calibration_error == pytest.approx(0.6, abs=1e-9)


def test_reliability_bins_cover_every_play() -> None:
    probability = np.linspace(0.0, 1.0, 500)
    truth = (probability > 0.5).astype(float)
    buckets, error = metrics.reliability(truth, probability, bins=10)
    assert sum(bucket.count for bucket in buckets) == 500
    assert 0.0 <= error <= 1.0


def _advisor(convert: float, make: float) -> fourth_down.FourthDownAdvisor:
    """An advisor whose only unknown is the arithmetic being tested."""
    return fourth_down.FourthDownAdvisor(
        win_probability=_LinearWinProbability(),
        conversion=_always(convert),
        field_goal=_always(make),
        punt=fourth_down.PuntModel({}, 75.0),
    )


def test_advisor_goes_for_it_on_fourth_and_inches() -> None:
    state = GameState(
        yardline_100=45.0, down=4, ydstogo=1.0, game_seconds_remaining=1800.0, score_differential=0
    )
    recommendation = _advisor(0.99, 0.5).recommend(state)
    assert recommendation.best.name == "go"
    assert recommendation.edge > 0
    assert "win probability points" in recommendation.render()


def test_advisor_kicks_when_the_conversion_is_hopeless() -> None:
    state = GameState(
        yardline_100=20.0, down=4, ydstogo=12.0, game_seconds_remaining=300.0, score_differential=-2
    )
    recommendation = _advisor(0.05, 0.95).recommend(state)
    assert sorted(option.name for option in recommendation.options) == sorted(fourth_down.OPTIONS)
    values = [option.win_probability for option in recommendation.options]
    assert values == sorted(values, reverse=True)
    assert recommendation.best.name == "field_goal"


def test_going_is_worth_more_the_likelier_the_conversion() -> None:
    state = GameState(
        yardline_100=40.0, down=4, ydstogo=3.0, game_seconds_remaining=900.0, score_differential=0
    )
    values = [
        next(
            option.win_probability
            for option in _advisor(convert, 0.5).recommend(state).options
            if option.name == "go"
        )
        for convert in (0.1, 0.5, 0.9)
    ]
    assert values == sorted(values)


def test_a_certain_conversion_is_worth_exactly_the_state_it_reaches() -> None:
    model = _LinearWinProbability()
    state = GameState(
        yardline_100=40.0, down=4, ydstogo=3.0, game_seconds_remaining=900.0, score_differential=0
    )
    go = next(
        option for option in _advisor(1.0, 0.5).recommend(state).options if option.name == "go"
    )
    reached = replace(
        state,
        down=1,
        ydstogo=3.0,
        yardline_100=37.0,
        goal_to_go=False,
        game_seconds_remaining=894.0,
        half_seconds_remaining=894.0,
    )
    assert go.win_probability == pytest.approx(model.probability(reached))


def test_punt_model_falls_back_when_a_bucket_is_thin() -> None:
    frame = pl.DataFrame(
        {"yardline_100": [60.0] * 30 + [20.0], "opponent_yardline_100": [78.0] * 30 + [95.0]}
    )
    model = fourth_down.PuntModel.fit(frame)
    assert model.opponent_yardline(60.0) == pytest.approx(78.0)
    assert model.opponent_yardline(20.0) == pytest.approx(78.0)


def test_punt_model_needs_punts() -> None:
    with pytest.raises(ValueError, match="no punts"):
        fourth_down.PuntModel.fit(pl.DataFrame({"yardline_100": [], "opponent_yardline_100": []}))


def test_play_call_model_beats_the_base_rate(plays: pl.DataFrame) -> None:
    dataset = features.split_by_season(
        plays,
        features.PLAYCALL_FEATURES,
        "is_pass_call",
        train=(2009, 2010),
        valid=(2019,),
        test=(2021,),
    )
    model = playcall.PlayCallModel.fit(dataset)
    fitted = metrics.score(
        features.target_vector(dataset.test.frame, dataset.target),
        model.predict(dataset.test.frame),
    )
    baseline = playcall.majority_baseline(dataset)
    assert fitted.log_loss < baseline.log_loss
    assert fitted.auc > baseline.auc


def test_play_call_round_trip(plays: pl.DataFrame, tmp_path) -> None:
    dataset = features.split_by_season(
        plays,
        features.PLAYCALL_FEATURES,
        "is_pass_call",
        train=(2009,),
        valid=(2010,),
        test=(2021,),
    )
    model = playcall.PlayCallModel.fit(dataset)
    path = tmp_path / "playcall.joblib"
    model.save(path)
    reloaded = playcall.PlayCallModel.load(path)
    np.testing.assert_allclose(
        reloaded.predict(dataset.test.frame), model.predict(dataset.test.frame)
    )


def test_predictability_rolls_up_to_team_seasons(plays: pl.DataFrame) -> None:
    dataset = features.split_by_season(
        plays,
        features.PLAYCALL_FEATURES,
        "is_pass_call",
        train=(2009, 2010),
        valid=(2019,),
        test=(2021,),
    )
    model = playcall.PlayCallModel.fit(dataset)
    report = playcall.predictability(model, dataset.test.frame, min_plays=10)
    assert report.rows
    assert all(0.0 <= row.predictability <= 1.0 for row in report.rows)
    assert all(0.5 <= row.confidence <= 1.0 for row in report.rows)
    assert {row.season for row in report.rows} == {2021}
    assert len(report.most_predictable(2)) == 2
    assert (
        report.most_predictable(1)[0].predictability
        >= report.least_predictable(1)[0].predictability
    )
    assert "predictability" in playcall.tendency_table(report.rows)


def test_predictability_handles_an_empty_frame(plays: pl.DataFrame) -> None:
    dataset = features.split_by_season(
        plays,
        features.PLAYCALL_FEATURES,
        "is_pass_call",
        train=(2009,),
        valid=(2010,),
        test=(2021,),
    )
    model = playcall.PlayCallModel.fit(dataset)
    report = playcall.predictability(model, dataset.test.frame.head(0))
    assert report.rows == ()


class _always:
    """A stand-in conversion or field-goal model with a fixed probability."""

    def __init__(self, value: float) -> None:
        self.value = value

    def probability(self, *_: object) -> float:
        return self.value


class _LinearWinProbability:
    """A closed-form stand-in for the network, so advisor arithmetic is checkable by hand."""

    def probability(self, state: GameState) -> float:
        value = 0.5 + 0.02 * state.score_differential + 0.004 * (100.0 - state.yardline_100)
        return float(np.clip(value, 0.01, 0.99))
