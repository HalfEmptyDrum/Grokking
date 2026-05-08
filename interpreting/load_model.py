import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import GrokModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str = None, weights_only_path: str = None,
               device: torch.device = None):
    """
    Recover a trained GrokModel.

    Pass either:
      - checkpoint_path: full checkpoint with config, history, optimizer state
      - weights_only_path: bare state_dict (requires you to know the config)

    Returns: (model, checkpoint_dict_or_None)
    """
    if device is None:
        device = get_device()

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = GrokModel(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt

    if weights_only_path is not None:
        # No config stored — assume defaults. Override here if you trained with different args.
        model = GrokModel().to(device)
        state_dict = torch.load(weights_only_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model, None

    raise ValueError("Provide either checkpoint_path or weights_only_path.")


def sanity_check(model: nn.Module, prime: int = 113, device: torch.device = None):
    """Evaluate the model on the full addition table and report accuracy."""
    if device is None:
        device = next(model.parameters()).device

    inputs = torch.tensor(
        [[a, b, prime] for a in range(prime) for b in range(prime)],
        dtype=torch.long, device=device,
    )
    targets = torch.tensor(
        [(a + b) % prime for a in range(prime) for b in range(prime)],
        dtype=torch.long, device=device,
    )

    with torch.no_grad():
        logits = model(inputs)[:, 2, :]                  # logits at "=" position
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()

    print(f"Full-table accuracy: {acc:.4f}  ({int(acc * len(targets))}/{len(targets)})")
    return acc


def predict(model: nn.Module, a: int, b: int, prime: int = 113,
            device: torch.device = None):
    """Predict a single (a + b) mod p."""
    if device is None:
        device = next(model.parameters()).device

    x = torch.tensor([[a, b, prime]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)[:, 2, :]
        pred = logits.argmax(dim=-1).item()
        prob = logits.softmax(dim=-1).max(dim=-1).values.item()

    truth = (a + b) % prime
    correct = "✓" if pred == truth else "✗"
    print(f"{a} + {b} mod {prime} = {truth} | model: {pred} (p={prob:.3f}) {correct}")
    return pred


if __name__ == "__main__":
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "model", "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "grok_checkpoint.pt")
    weights_path = os.path.join(ckpt_dir, "grok_model.pt")

    if os.path.exists(ckpt_path):
        print(f"Loading full checkpoint: {ckpt_path}")
        model, ckpt = load_model(checkpoint_path=ckpt_path)
        print(f"Trained for {ckpt['step']} steps.")
        if "history" in ckpt and ckpt["history"]["test_acc"]:
            print(f"Final logged test acc: {ckpt['history']['test_acc'][-1]:.4f}")
    else:
        print(f"Loading weights only: {weights_path}")
        model, _ = load_model(weights_only_path=weights_path)

    sanity_check(model)

    # spot-check a few examples
    predict(model, 42, 97)
    predict(model, 0, 0)
    predict(model, 112, 1)
    predict(model, 56, 56)