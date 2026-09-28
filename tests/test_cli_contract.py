"""The CLI, the Makefile and the YAML configs must agree with each other.

These are not tests of behaviour -- they are drift detectors. Every one of them exists
because the three surfaces got out of sync at least once: the Makefile called subcommands
that had been renamed, and base.yaml declared keys that no code path ever read. A config
key nobody reads is worse than a missing one: you change it, nothing happens, and you
spend an evening wondering why the run did not respond.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from qheart.cli import CONFIG_MAP, _apply_config, _dig, build_parser

ROOT = Path(__file__).resolve().parents[1]


def _subparsers(parser):
    act = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    return act.choices


def _all_dests(parser):
    return {d for sp in _subparsers(parser).values() for d in (a.dest for a in sp._actions)}


def _configs():
    yaml = pytest.importorskip("yaml")
    return {p: yaml.safe_load(p.read_text()) or {}
            for p in sorted(ROOT.joinpath("configs").rglob("*.yaml"))}


# --------------------------------------------------------------- Makefile vs CLI
def test_every_makefile_invocation_names_a_real_subcommand():
    mk = ROOT.joinpath("Makefile").read_text()
    called = set(re.findall(r"qheart\.cli\s+([a-z_]+)", mk))
    real = set(_subparsers(build_parser()))
    assert called, "found no qheart.cli invocations in the Makefile -- did it move?"
    missing = sorted(called - real)
    assert not missing, (
        f"Makefile calls subcommands that do not exist: {missing}. "
        f"Real subcommands: {sorted(real)}"
    )


def test_makefile_passes_config_only_where_config_is_accepted():
    mk = ROOT.joinpath("Makefile").read_text()
    parsers = _subparsers(build_parser())
    for cmd, tail in re.findall(r"qheart\.cli\s+([a-z_]+)([^\n]*)", mk):
        if "--config" not in tail:
            continue
        sp = parsers[cmd]
        assert any(a.dest == "config" for a in sp._actions), \
            f"make passes --config to `{cmd}`, which does not accept it"


# --------------------------------------------------------------- CLI vs configs
def test_every_subcommand_accepts_config():
    for name, sp in _subparsers(build_parser()).items():
        assert any(a.dest == "config" for a in sp._actions), \
            f"subcommand `{name}` cannot be driven from a config file"


def test_no_config_mapping_points_at_a_flag_that_does_not_exist():
    dests = _all_dests(build_parser())
    orphans = sorted(f"{k} -> {v}" for k, v in CONFIG_MAP.items() if v not in dests)
    assert not orphans, (
        f"these config keys map to arguments no subcommand defines, so setting them "
        f"in YAML does nothing: {orphans}"
    )


def test_no_config_mapping_is_dead():
    cfgs = _configs()
    dead = [k for k in CONFIG_MAP if not any(_dig(c, k)[1] for c in cfgs.values())]
    assert not dead, f"CONFIG_MAP keys that appear in no config file: {sorted(dead)}"


def test_base_config_declares_the_seed_once_and_only_once():
    cfgs = _configs()
    base = ROOT / "configs" / "base.yaml"
    assert _dig(cfgs[base], "seed") == (20260830, True)
    others = [p.name for p, c in cfgs.items() if p != base and "seed" in c]
    assert not others, f"seed re-declared outside base.yaml: {others} -- one seed, one place"


# --------------------------------------------------------------- precedence
def test_explicit_flag_beats_the_config_file():
    """Otherwise a config silently overrides what someone typed, which costs an hour."""
    parser = build_parser()
    args = parser.parse_args(["params", "--config", "configs/base.yaml", "--n-qubits", "7"])
    _apply_config(args, parser)
    assert args.n_qubits == 7


def test_config_fills_in_what_the_caller_left_alone(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"quantum": {"n_qubits": 6, "depth": 3}}))
    parser = build_parser()
    args = parser.parse_args(["params", "--config", str(cfg)])
    applied = _apply_config(args, parser)
    assert (args.n_qubits, args.depth) == (6, 3)
    assert len(applied) == 2


def test_a_config_key_nobody_maps_is_ignored_rather_than_crashing(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"quantum": {"n_qubits": 6}, "nonsense": {"x": 1}}))
    parser = build_parser()
    args = parser.parse_args(["params", "--config", str(cfg)])
    _apply_config(args, parser)
    assert args.n_qubits == 6


def test_no_config_means_no_overlay():
    parser = build_parser()
    args = parser.parse_args(["params"])
    assert _apply_config(args, parser) == []
