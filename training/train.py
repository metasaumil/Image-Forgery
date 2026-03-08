"""
training/train.py
-----------------
Main training script. Run this to train your CNN.

Example usage:
    # With dummy data (for testing)
    python training/train.py --model resnet18 --epochs 5 --dummy

    # With real data
    python training/train.py --model resnet18 --data_dir data/processed --epochs 30

    # Lightweight model
    python training/train.py --model mobilenetv2 --batch_size 64 --lr 1e-3
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning.models       import get_model
from preprocessing.dataset      import get_dataloaders
from preprocessing.prepare_data import create_dummy_dataset
from training.trainer           import train, plot_history
from evaluation.evaluate        import full_evaluation


def main(args):
    # ── Setup ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Image Forgery Detection — Training")
    print(f"  Model:   {args.model}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"{'='*60}\n")

    # ── Data ───────────────────────────────────────────────
    if args.dummy:
        print("[Setup] Creating dummy dataset for quick testing...")
        create_dummy_dataset(args.data_dir, n_per_class=args.dummy_n)

    print(f"[Setup] Loading data from: {args.data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )

    # ── Model ──────────────────────────────────────────────
    model = get_model(args.model, num_classes=2, pretrained=args.pretrained)

    # ── Train ──────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = train(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        num_epochs      = args.epochs,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        patience        = args.patience,
        checkpoint_dir  = args.checkpoint_dir,
        model_name      = args.model,
        device          = device
    )

    # ── Plot ───────────────────────────────────────────────
    plot_history(history, save_path=os.path.join(args.checkpoint_dir, f"{args.model}_history.png"))

    # ── Evaluate ───────────────────────────────────────────
    if args.evaluate:
        print("\n[Evaluate] Loading best model for evaluation...")
        checkpoint = os.path.join(args.checkpoint_dir, f"best_{args.model}.pth")
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        full_evaluation(model, test_loader,
                        output_dir=f"evaluation/results",
                        model_name=args.model,
                        device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image Forgery Detector")

    # Data
    parser.add_argument("--data_dir",    default="data/processed", help="Root data directory")
    parser.add_argument("--dummy",       action="store_true",       help="Generate dummy data")
    parser.add_argument("--dummy_n",     type=int, default=100,     help="Images per class for dummy")
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--model",       default="resnet18",
                        choices=["custom", "resnet18", "mobilenetv2"])
    parser.add_argument("--pretrained",  action="store_true", default=True)

    # Training
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=7)
    parser.add_argument("--checkpoint_dir", default="models")

    # Evaluation
    parser.add_argument("--evaluate", action="store_true", default=True)

    args = parser.parse_args()
    main(args)
