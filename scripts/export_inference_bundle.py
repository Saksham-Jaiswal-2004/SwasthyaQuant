#!/usr/bin/env python3
"""Export the frozen inference bundle expected by the FastAPI backend.

Usage examples:

1. Export from a Python module that already contains the trained objects:
    python scripts/export_inference_bundle.py \
        --module qheart.training \
        --clinical-representation clinical_representation \
        --selector selector \
        --dcqf dcqf \
        --scaler scaler \
        --model model

2. Export from a bundle object already defined in a module:
    python scripts/export_inference_bundle.py \
        --module qheart.training \
        --bundle-var inference_bundle

The output will be written to backend/artifacts/inference_bundle.joblib by default,
which matches the backend lookup contract in backend/app/config.py.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "artifacts" / "inference_bundle.joblib"


def _load_object(module_name: str, attribute_name: str):
    module = import_module(module_name)
    if not hasattr(module, attribute_name):
        raise AttributeError(f"Module '{module_name}' has no attribute '{attribute_name}'.")
    return getattr(module, attribute_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        required=True,
        help="Python import path for the module containing the trained objects, e.g. my_project.training",
    )
    parser.add_argument(
        "--bundle-var",
        help="Name of a dict variable already containing the inference bundle in the module.",
    )
    parser.add_argument(
        "--clinical-representation",
        help="Name of the fitted ClinicalRepresentation object in the module.",
    )
    parser.add_argument(
        "--selector",
        help="Name of the fitted feature selector object in the module.",
    )
    parser.add_argument(
        "--dcqf",
        help="Name of the fitted DCQFExtractor object in the module.",
    )
    parser.add_argument(
        "--scaler",
        help="Name of the fitted StandardScaler object in the module.",
    )
    parser.add_argument(
        "--model",
        help="Name of the fitted classifier object in the module.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination for the serialized bundle (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def resolve_bundle(module_name: str, args: argparse.Namespace) -> dict[str, object]:
    if args.bundle_var:
        bundle = _load_object(module_name, args.bundle_var)
        if not isinstance(bundle, dict):
            raise TypeError(f"{args.bundle_var} must resolve to a dict, got {type(bundle).__name__}.")
        return bundle

    required = [
        ("clinical_representation", args.clinical_representation),
        ("selector", args.selector),
        ("dcqf", args.dcqf),
        ("scaler", args.scaler),
        ("model", args.model),
    ]
    bundle: dict[str, object] = {}
    for key, name in required:
        if name is None:
            raise ValueError(
                f"Missing required object name for '{key}'. "
                "Provide --bundle-var or all of --clinical-representation --selector --dcqf --scaler --model."
            )
        bundle[key] = _load_object(module_name, name)
    return bundle


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    bundle = resolve_bundle(args.module, args)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    print(f"Saved inference bundle to {output_path}")
    print(f"Keys: {', '.join(sorted(bundle))}")


if __name__ == "__main__":
    main()
