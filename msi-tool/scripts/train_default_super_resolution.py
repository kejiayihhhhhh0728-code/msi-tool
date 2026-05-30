from __future__ import annotations

import argparse
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.super_resolution import train_and_evaluate_vdsr


DEFAULT_PATCH_ROOT = r"E:\Study\竞赛-代码\超分辨\20个样本\20个样本\CUT-ge"
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "super_resolution_models")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train default MSI super-resolution model on CUT-ge patches.")
    parser.add_argument("--patch-root", default=DEFAULT_PATCH_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default="lcrn_guided", choices=["vanilla", "he_guided", "lcrn_guided"])
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fresh", action="store_true", help="Do not resume from an existing checkpoint.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Default SR training")
    print(f"Patch root : {args.patch_root}")
    print(f"Output dir : {args.output_dir}")
    print(f"Variant    : {args.variant}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.learning_rate}")
    print("-" * 60, flush=True)

    result = train_and_evaluate_vdsr(
        patch_root=args.patch_root,
        output_dir=args.output_dir,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=not args.fresh,
    )

    default_name = f"{args.variant}_default.pth"
    default_path = os.path.join(args.output_dir, default_name)
    shutil.copyfile(result["model_path"], default_path)
    print("-" * 60)
    print(f"Training complete: {result['model_path']}")
    print(f"Default model copied to: {default_path}")
    print(f"Metrics CSV: {result['metrics_csv']}")
    print(f"Average metrics: {result['model_average']}")


if __name__ == "__main__":
    main()
