import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import GrokModel
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.generate_dataset import split_training_test, generate_dataset


def build_tensors(data, prime):
    inputs = torch.tensor([[a, b, prime] for a, b, _ in data], dtype=torch.long)
    targets = torch.tensor([r for _, _, r in data], dtype=torch.long)
    return inputs, targets


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(prime: int = 113):
    device = get_device()
    print(f"Using device: {device}")

    model = GrokModel().to(device)
    dataset = generate_dataset()
    train_data, test_data = split_training_test(dataset)

    train_inputs, train_targets = build_tensors(train_data, prime)
    test_inputs, test_targets = build_tensors(test_data, prime)
    train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
    test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    loss_fn = nn.CrossEntropyLoss()

    num_steps = 40_000
    batch_size = 512
    log_every = 100

    history = {"step": [], "train_loss": [], "test_loss": [],
               "train_acc": [], "test_acc": []}

    plt.ion()
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    for step in range(num_steps):
        model.train()
        idx = torch.randint(0, len(train_inputs), (batch_size,), device=device)
        x, y = train_inputs[idx], train_targets[idx]

        optimizer.zero_grad()
        logits = model(x)[:, 2, :]                # <- slice the "=" position
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        if step > 0 and step % log_every == 0:
            with torch.no_grad():
                model.eval()
                train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
                test_logits = model(test_inputs)[:, 2, :]
                test_loss = loss_fn(test_logits, test_targets).item()
                test_acc = (test_logits.argmax(dim=-1) == test_targets).float().mean().item()

            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            print(f"Step {step:6d} | "
                  f"Train loss {loss.item():.4f} acc {train_acc:.3f} | "
                  f"Test loss {test_loss:.4f} acc {test_acc:.3f}")

            # --- live update ---
            ax_loss.clear()
            ax_loss.plot(history["step"], history["train_loss"], label="train")
            ax_loss.plot(history["step"], history["test_loss"], label="test")
            ax_loss.set_xscale("log"); ax_loss.set_yscale("log")
            ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
            ax_loss.set_title("Loss"); ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)

            ax_acc.clear()
            ax_acc.plot(history["step"], history["train_acc"], label="train")
            ax_acc.plot(history["step"], history["test_acc"], label="test")
            ax_acc.set_xscale("log")
            ax_acc.set_xlabel("step"); ax_acc.set_ylabel("accuracy")
            ax_acc.set_title("Accuracy"); ax_acc.set_ylim(-0.05, 1.05)
            ax_acc.legend(); ax_acc.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    train_model()