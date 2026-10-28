"""Declarative study plans (#112): a validated spec of tasks, arms, seeds and tuning grids, expanded into units.

A unit is (phase, task, arm, seed, value): the unit of record, as in #24. The spec is plain data (a YAML file
validated at load), so a predeclaration can reference it byte for byte:

    study: pendulum-ccil
    version: 5
    tasks: [continuous, discontinuous]
    settings: [transition_model.trainer_version=2, policy.trainer_version=2]   # every trained arm
    phases:
      tuning: {seeds: [1, 2, 3], arms: [noise_bc, ccil_tuned], tuning: true}
      study: {seeds: {from: 101, to: 120}, arms: all}
    arms:
      bc: {settings: [policy.method=bc], noise_levels: [0.0, 0.01, 0.05, 0.1]}
      noise_bc: {settings: [policy.method=noise_bc], tuned: policy.noise_scale}
      bc_frac10: {settings: [policy.method=bc], data_fraction: 0.1}
      zero: {train: false, noise_levels: [0.0, 0.01, 0.05, 0.1]}
    tuning:
      noise_bc: {grid: [1.0e-4, 1.0e-3], select: highest_mean_noise_free_R, ties: smaller}
      ccil_tuned: {grid: {continuous: [1.0e-5, 1.0e-3], discontinuous: [1.0e-4, 1.0e-3]}, ...}

In the tuning phase a tuned arm runs every grid value; in later phases it runs the value the choice selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SELECT_RULES = ("highest_mean_noise_free_R", "lowest_mean_validation_loss")
TIE_RULES = ("smaller",)


class PlanError(ValueError):
    """The study spec is not valid; the message lists every problem."""


@dataclass(frozen=True)
class Arm:
    name: str
    settings: tuple[str, ...] = ()
    noise_levels: tuple[float, ...] = (0.0,)
    data_fraction: float | None = None
    tuned: str | None = None          # the configuration key a tuning grid sets
    train: bool = True                # false: a reference with no training (for example zero action)


@dataclass(frozen=True)
class Tuning:
    grid: dict                        # task -> tuple of values
    select: str = SELECT_RULES[0]
    ties: str = TIE_RULES[0]


@dataclass(frozen=True)
class Study:
    name: str
    version: int
    tasks: tuple[str, ...]
    phases: dict                      # phase -> {"seeds": tuple, "arms": tuple}
    arms: dict                        # name -> Arm
    tuning: dict = field(default_factory=dict)  # tuned arm -> Tuning
    settings: tuple[str, ...] = ()    # configuration overrides of every trained arm, before the arm's own

    def deferred(self, phase: str) -> tuple[str, ...]:
        """Arms whose returns a phase must not compute: in a tuning phase, those selected by validation loss. Their
        returns come after the choice is recorded (runner.evaluate_deferred), so no return can inform it."""
        if not self.phases[phase]["tuning"]:
            return ()
        return tuple(a for a in self.phases[phase]["arms"]
                     if a in self.tuning and self.tuning[a].select == "lowest_mean_validation_loss")

    def unit_settings(self, arm: str, value=None) -> list[str]:
        """The overrides a trained unit runs with: the study's, the arm's, then the tuned value."""
        a = self.arms[arm]
        return [*self.settings, *a.settings, *([f"{a.tuned}={value!r}"] if a.tuned and value is not None else [])]

    def units(self, phase: str, choice: dict | None = None) -> list[tuple]:
        """Every (task, arm, seed, value) of a phase, in a fixed order: tasks, then seeds, then arms."""
        if phase not in self.phases:
            raise PlanError([f"phase {phase!r} is not one of {sorted(self.phases)}"])
        spec, out = self.phases[phase], []
        tuning_phase = spec["tuning"]
        for task in self.tasks:
            for seed in spec["seeds"]:
                for arm in spec["arms"]:
                    a = self.arms[arm]
                    if a.tuned and tuning_phase:
                        out += [(task, arm, seed, v) for v in self.tuning[arm].grid[task]]
                    elif a.tuned:
                        if choice is None:
                            raise PlanError([f"phase {phase!r} runs tuned arm {arm!r}: it needs the tuning choice"])
                        out.append((task, arm, seed, choice[task][arm]["value"]))
                    else:
                        out.append((task, arm, seed, None))
        return out


def _seeds(v, where, problems):
    if isinstance(v, dict) and set(v) == {"from", "to"} and all(isinstance(x, int) for x in v.values()):
        return tuple(range(v["from"], v["to"] + 1))
    if isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        if len(set(v)) != len(v):
            problems.append(f"{where}: seeds repeat")
        return tuple(v)
    problems.append(f"{where}: seeds must be a non-empty list of integers or {{from, to}}")
    return ()


def from_dict(d: dict) -> Study:
    """Validate a spec and build the Study; PlanError lists every problem."""
    problems = []
    known = {"study", "version", "tasks", "phases", "arms", "tuning", "settings"}
    for k in set(d) - known:
        problems.append(f"unknown key {k!r}")
    for k in ("study", "version", "tasks", "phases", "arms"):
        if k not in d:
            problems.append(f"missing key {k!r}")
    if problems:
        raise PlanError(problems)
    tasks = tuple(d["tasks"])
    common = d.get("settings") or []
    if not isinstance(common, list) or not all(isinstance(x, str) and "=" in x for x in common):
        problems.append("settings must be a list of key=value overrides")
    arms = {}
    for name, a in d["arms"].items():
        extra = set(a) - {"settings", "noise_levels", "data_fraction", "tuned", "train"}
        if extra:
            problems.append(f"arm {name}: unknown keys {sorted(extra)}")
        frac = a.get("data_fraction")
        if frac is not None and not 0 < frac <= 1:
            problems.append(f"arm {name}: data_fraction must be in (0, 1]")
        if not a.get("train", True) and (a.get("settings") or a.get("tuned")):
            problems.append(f"arm {name}: an arm with no training takes no settings or tuning")
        arms[name] = Arm(name, tuple(a.get("settings", ())), tuple(float(x) for x in a.get("noise_levels", [0.0])),
                         frac, a.get("tuned"), bool(a.get("train", True)))
    tuning = {}
    for name, t in (d.get("tuning") or {}).items():
        if name not in arms or not arms[name].tuned:
            problems.append(f"tuning {name}: not a tuned arm")
            continue
        grid = t.get("grid")
        grid = {task: tuple(grid) for task in tasks} if isinstance(grid, list) else (grid or {})
        if set(grid) != set(tasks) or any(not g for g in grid.values()):
            problems.append(f"tuning {name}: the grid needs a non-empty list for every task")
        if t.get("select", SELECT_RULES[0]) not in SELECT_RULES or t.get("ties", TIE_RULES[0]) not in TIE_RULES:
            problems.append(f"tuning {name}: select must be one of {SELECT_RULES}, ties one of {TIE_RULES}")
        tuning[name] = Tuning({k: tuple(float(x) for x in v) for k, v in grid.items()},
                              t.get("select", SELECT_RULES[0]), t.get("ties", TIE_RULES[0]))
    for name, a in arms.items():
        if a.tuned and name not in tuning:
            problems.append(f"arm {name}: tuned by {a.tuned!r} but has no tuning grid")
    phases = {}
    for phase, p in d["phases"].items():
        seeds = _seeds(p.get("seeds"), f"phase {phase}", problems)
        names = tuple(arms) if p.get("arms") == "all" else tuple(p.get("arms") or ())
        missing = [a for a in names if a not in arms]
        if missing:
            problems.append(f"phase {phase}: unknown arms {missing}")
        if p.get("tuning") and any(not arms[a].tuned for a in names if a in arms):
            problems.append(f"phase {phase}: a tuning phase runs tuned arms only")
        phases[phase] = {"seeds": seeds, "arms": names, "tuning": bool(p.get("tuning", False))}
    seen = {}
    for phase, p in phases.items():
        for s in p["seeds"]:
            if s in seen:
                problems.append(f"seed {s} is in phases {seen[s]!r} and {phase!r}; phases must not share seeds")
            seen[s] = phase
    if problems:
        raise PlanError(problems)
    return Study(str(d["study"]), int(d["version"]), tasks, phases, arms, tuning, tuple(common))


def load(path) -> Study:
    """A Study from a YAML spec file."""
    return from_dict(yaml.safe_load(Path(path).read_text()))
