"""
Regression test for orchestrator scripts that subprocess train_client.py.

The class of bug this catches: an orchestrator passes --some-flag with its
own default that DIFFERS from train_client.py's default for the same flag,
silently overriding it. This is what happened with --checkpoint-every (03
defaulted to 0, train_client to 5 -> no periodic checkpoints were saved).

For each orchestrator we list the flags it forwards to train_client.py and
assert that the orchestrator's default equals train_client.py's default.
The orchestrator may legitimately override a default (e.g. --patience=15
for the federated-experiment baseline) — those go in PERMITTED_OVERRIDES.

Run: python test_orchestrator_defaults.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_module_from_path(path: Path, name: str):
    """Load a .py file as a module even if its filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_with_argv(parse_args_fn, argv: list[str]):
    """Call parse_args_fn with sys.argv temporarily set to argv."""
    saved = sys.argv
    sys.argv = ["test"] + argv
    try:
        return parse_args_fn()
    finally:
        sys.argv = saved


# Required args so each parser doesn't bail on missing arguments.
TRAIN_CLIENT_REQUIRED = [
    "--client-id", "0",
    "--fold",      "1",
    "--model",     "facebook/bart-base",
    "--family",    "bart",
    "--targets",   "q_proj", "v_proj",
]
ORCH_03_REQUIRED = ["--splits-dir", "x", "--output-dir", "y"]
ORCH_07_REQUIRED = ["--splits-dir", "x", "--output-dir", "y"]


# (orchestrator dest -> train_client dest) for flags the orchestrator forwards
# verbatim. If the orchestrator and train_client use different dest names, list
# the pair; otherwise the same name appears twice.
PASSTHROUGHS_03 = {
    "num_epochs":       "num_epochs",
    "batch_size":       "batch_size",
    "lr":               "lr",
    "patience":         "patience",
    "preview_every":    "preview_every",
    "checkpoint_every": "checkpoint_every",
}
PASSTHROUGHS_07 = {
    "num_epochs":     "num_epochs",
    "batch_size":     "batch_size",
    "lr":             "lr",
    "warmup_ratio":   "warmup_ratio",
    "grad_clip":      "grad_clip",
    "lora_r":         "lora_r",
    "lora_alpha":     "lora_alpha",
    "lora_dropout":   "lora_dropout",
    "max_input_len":  "max_input_len",
    "max_target_len": "max_target_len",
    # NOTE: 07 deliberately overrides patience (15 vs train_client's 10) for
    # federated-experiment baselines — listed in PERMITTED_OVERRIDES.
    "patience":       "patience",
}

# Orchestrator flags whose default is INTENTIONALLY different from train_client.
# Format: (orchestrator_module_name, dest) -> (orch_default, tc_default).
# Adding an entry here documents that the divergence is on purpose.
PERMITTED_OVERRIDES = {
    ("orch_07", "patience"): (15, 10),
}


def _check_one(label: str, orch_args, tc_args, mapping: dict, module_key: str) -> list[str]:
    failures = []
    for orch_dest, tc_dest in mapping.items():
        if not hasattr(orch_args, orch_dest):
            failures.append(
                f"  [{label}] orchestrator missing flag dest={orch_dest!r}"
            )
            continue
        if not hasattr(tc_args, tc_dest):
            failures.append(
                f"  [{label}] train_client missing flag dest={tc_dest!r}"
            )
            continue
        ov = getattr(orch_args, orch_dest)
        tv = getattr(tc_args, tc_dest)
        if ov == tv:
            continue
        permitted = PERMITTED_OVERRIDES.get((module_key, orch_dest))
        if permitted == (ov, tv):
            print(f"  [{label}] OK (permitted override) "
                  f"{orch_dest}: {ov} != train_client {tc_dest}={tv}")
            continue
        failures.append(
            f"  [{label}] DEFAULT MISMATCH: orchestrator {orch_dest}={ov!r} "
            f"vs train_client {tc_dest}={tv!r}. Either align defaults or add "
            f"({module_key!r}, {orch_dest!r}): ({ov!r}, {tv!r}) to "
            f"PERMITTED_OVERRIDES."
        )
    return failures


def _check_no_load_adapter_calls() -> list[str]:
    """
    Guard against the PEFT footgun: model.load_adapter("default") creates a
    NEW adapter rather than overwriting the existing one. Any LoRA round-trip
    must use set_peft_model_state_dict() instead. Comments mentioning
    load_adapter are fine; actual call sites are not.
    """
    import re
    failures: list[str] = []
    # Match `.load_adapter(` after an identifier, but not inside a # comment.
    call_re = re.compile(r"^[^#\n]*\b\w+\.load_adapter\s*\(", re.MULTILINE)
    for path in (REPO / "unifiedfl").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in call_re.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            failures.append(
                f"  [load_adapter guard] {path.relative_to(REPO)}:{line_no} "
                f"calls .load_adapter(...) — use set_peft_model_state_dict instead."
            )
    return failures


def _load_all():
    train_client = _load_module_from_path(
        REPO / "unifiedfl" / "train_client.py", "train_client"
    )
    orch_03 = _load_module_from_path(
        REPO / "experiments" / "03_run_three_conditionings.py", "orch_03"
    )
    orch_07 = _load_module_from_path(
        REPO / "experiments" / "07_run_individual_baselines_fed_exp.py", "orch_07"
    )
    tc_args = _parse_with_argv(train_client.parse_args, TRAIN_CLIENT_REQUIRED)
    a03    = _parse_with_argv(orch_03.parse_args,    ORCH_03_REQUIRED)
    a07    = _parse_with_argv(orch_07.parse_args,    ORCH_07_REQUIRED)
    return tc_args, a03, a07


# ── pytest entry points ──────────────────────────────────────────────────────

def test_03_passthrough_defaults_match_train_client():
    tc_args, a03, _ = _load_all()
    fails = _check_one("03 vs train_client", a03, tc_args, PASSTHROUGHS_03, "orch_03")
    assert not fails, "\n".join(fails)


def test_07_passthrough_defaults_match_train_client():
    tc_args, _, a07 = _load_all()
    fails = _check_one("07 vs train_client", a07, tc_args, PASSTHROUGHS_07, "orch_07")
    assert not fails, "\n".join(fails)


def test_no_load_adapter_calls_in_unifiedfl():
    fails = _check_no_load_adapter_calls()
    assert not fails, "\n".join(fails)


# ── script entry point (legacy `python tests/test_orchestrator_defaults.py`) ─

def main() -> int:
    tc_args, a03, a07 = _load_all()
    failures: list[str] = []
    failures += _check_one("03 vs train_client", a03, tc_args, PASSTHROUGHS_03, "orch_03")
    failures += _check_one("07 vs train_client", a07, tc_args, PASSTHROUGHS_07, "orch_07")
    failures += _check_no_load_adapter_calls()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f)
        print(f"\n{len(failures)} mismatch(es) found.")
        return 1
    print("\nAll orchestrator passthrough defaults match train_client.py.")
    print("No stray .load_adapter() call sites under unifiedfl/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
