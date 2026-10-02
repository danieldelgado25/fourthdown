"""One pass/fail verdict over every evaluation suite, against checked-in thresholds.

Each harness already writes its own report; the scorecard is what lets CI act on them.
Every gated metric has a floor or a ceiling in ``gates.json`` with the reason it sits
there, so a regression fails the build with a sentence explaining why it matters instead
of a number nobody remembers the meaning of.

A ceiling is as useful as a floor. A win-probability AUC of 0.97 on held-out seasons is
not a breakthrough, it is a leaked feature, and the gate says so.

Suites that cannot run (no Ollama, no Postgres) are skipped rather than failed, unless
the caller marks them required -- which is how CI makes sure "skipped" never silently
stands in for "passed".
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources

GATES_FILE = "gates.json"


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Gate:
    """A bound on one metric, keyed ``suite.metric``."""

    key: str
    minimum: float | None
    maximum: float | None
    why: str

    @property
    def suite(self) -> str:
        return self.key.partition(".")[0]

    @property
    def metric(self) -> str:
        return self.key.partition(".")[2]

    def admits(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.minimum is not None and value < self.minimum:
            return False
        return self.maximum is None or value <= self.maximum

    def bound(self) -> str:
        if self.minimum is not None and self.maximum is not None:
            return f"{self.minimum:g} to {self.maximum:g}"
        if self.minimum is not None:
            return f">= {self.minimum:g}"
        return f"<= {self.maximum:g}"


@dataclass(frozen=True)
class SuiteResult:
    """What one suite measured, or why it could not run."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    findings: tuple[str, ...] = ()
    skipped: str | None = None
    seconds: float = 0.0


@dataclass(frozen=True)
class Verdict:
    gate: Gate
    value: float | None
    status: Status


def load_gates() -> tuple[Gate, ...]:
    raw = resources.files("fourthdown.evaluation").joinpath(GATES_FILE).read_text()
    return tuple(
        Gate(key=key, minimum=spec.get("min"), maximum=spec.get("max"), why=spec["why"])
        for key, spec in json.loads(raw).items()
    )


def _format(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}".rstrip("0").rstrip(".") if not math.isnan(value) else "nan"


@dataclass(frozen=True)
class Scorecard:
    suites: tuple[SuiteResult, ...]
    gates: tuple[Gate, ...]
    required: frozenset[str] = frozenset()

    def _suite(self, name: str) -> SuiteResult | None:
        return next((suite for suite in self.suites if suite.name == name), None)

    def verdicts(self) -> list[Verdict]:
        """One verdict per gate whose suite was asked for.

        A suite that ran but did not produce a gated metric fails: a renamed metric must
        not quietly turn a gate off.
        """
        verdicts = []
        for gate in self.gates:
            suite = self._suite(gate.suite)
            if suite is None:
                continue
            if suite.skipped is not None:
                verdicts.append(Verdict(gate, None, Status.SKIPPED))
                continue
            value = suite.metrics.get(gate.metric)
            if value is None:
                verdicts.append(Verdict(gate, None, Status.FAIL))
            else:
                verdicts.append(
                    Verdict(gate, value, Status.PASS if gate.admits(value) else Status.FAIL)
                )
        return verdicts

    def missing_required(self) -> list[str]:
        """Required suites that were skipped or never run at all."""
        missing = []
        for name in sorted(self.required):
            suite = self._suite(name)
            if suite is None or suite.skipped is not None:
                missing.append(name)
        return missing

    def failures(self) -> list[Verdict]:
        return [verdict for verdict in self.verdicts() if verdict.status is Status.FAIL]

    @property
    def passed(self) -> bool:
        return not self.failures() and not self.missing_required()

    def to_json(self) -> str:
        return json.dumps(
            {
                "passed": self.passed,
                "missing_required": self.missing_required(),
                "suites": [
                    {
                        "name": suite.name,
                        "skipped": suite.skipped,
                        "seconds": round(suite.seconds, 1),
                        "metrics": suite.metrics,
                        "findings": list(suite.findings),
                    }
                    for suite in self.suites
                ],
                "gates": [
                    {
                        "key": verdict.gate.key,
                        "min": verdict.gate.minimum,
                        "max": verdict.gate.maximum,
                        "value": verdict.value,
                        "status": verdict.status.value,
                    }
                    for verdict in self.verdicts()
                ],
            },
            indent=2,
        )

    def render(self) -> str:
        verdicts = self.verdicts()
        gated = {verdict.gate.key for verdict in verdicts}
        headline = "PASS" if self.passed else "FAIL"
        lines = [
            "# Evaluation scorecard",
            "",
            f"**{headline}**: {sum(v.status is Status.PASS for v in verdicts)} gates passed, "
            f"{len(self.failures())} failed, "
            f"{sum(v.status is Status.SKIPPED for v in verdicts)} skipped.",
        ]
        missing = self.missing_required()
        if missing:
            lines.append(f"Required suites that did not run: {', '.join(missing)}.")
        lines.extend(["", "## Suites", "", "| suite | status | time |", "| --- | --- | --- |"])
        for suite in self.suites:
            status = f"skipped: {suite.skipped}" if suite.skipped else "ran"
            lines.append(f"| {suite.name} | {status} | {suite.seconds:.0f}s |")
        lines.extend(
            [
                "",
                "## Gates",
                "",
                "| metric | value | bound | status | why |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for verdict in verdicts:
            lines.append(
                f"| {verdict.gate.key} | {_format(verdict.value)} | {verdict.gate.bound()} "
                f"| {verdict.status.value} | {verdict.gate.why} |"
            )
        informational = [
            (suite.name, name, value)
            for suite in self.suites
            for name, value in suite.metrics.items()
            if f"{suite.name}.{name}" not in gated
        ]
        if informational:
            lines.extend(["", "## Reported, not gated", "", "| metric | value |", "| --- | --- |"])
            lines.extend(
                f"| {suite}.{name} | {_format(value)} |" for suite, name, value in informational
            )
        findings = [(suite.name, finding) for suite in self.suites for finding in suite.findings]
        if findings:
            lines.extend(["", "## Findings", ""])
            lines.extend(f"- **{suite}**: {finding}" for suite, finding in findings)
        return "\n".join(lines) + "\n"


def build(
    suites: Iterable[SuiteResult],
    *,
    required: Sequence[str] = (),
    gates: Sequence[Gate] | None = None,
) -> Scorecard:
    return Scorecard(
        suites=tuple(suites),
        gates=tuple(gates) if gates is not None else load_gates(),
        required=frozenset(required),
    )
