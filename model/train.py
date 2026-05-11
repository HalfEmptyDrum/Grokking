import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import GrokModel, SimpleModel
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


def train_simple_model(prime: int = 113):
    device = get_device()
    print(f"Using device: {device}")

    model = SimpleModel().to(device)
    dataset = generate_dataset()
    train_data, test_data = split_training_test(dataset)

    train_inputs, train_targets = build_tensors(train_data, prime)
    test_inputs, test_targets = build_tensors(test_data, prime)
    train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
    test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.5)
    loss_fn = nn.CrossEntropyLoss()

    num_steps = 160_000
    batch_size = 128
    log_every = 50

    history = {"step": [], "train_loss": [], "test_loss": [],
           "train_acc": [], "test_acc": [],
           "w1_norm": [], "w2_norm": [], "total_norm": []}

    plt.ion()
    fig, (ax_loss, ax_acc, ax_norm) = plt.subplots(1, 3, figsize=(16, 4))

    for step in range(num_steps):
        model.train()
        idx = torch.randint(0, len(train_inputs), (batch_size,), device=device)
        x, y = train_inputs[idx], train_targets[idx]
        a = x[:, 0]
        b = x[:, 1]
        optimizer.zero_grad()
        logits = model(a, b)                # <- slice the "=" position
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        if step > 0 and step % log_every == 0:
            with torch.no_grad():
                model.eval()
                train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
                test_logits = model(test_inputs[:, 0], test_inputs[:, 1])
                test_loss = loss_fn(test_logits, test_targets).item()
                test_acc = (test_logits.argmax(dim=-1) == test_targets).float().mean().item()

                w1_norm = model.fc1.weight.norm().item()
                w2_norm = model.fc2.weight.norm().item()
                total_norm = sum(p.norm().item()**2 for p in model.parameters())**0.5

            history["w1_norm"].append(w1_norm)
            history["w2_norm"].append(w2_norm)
            history["total_norm"].append(total_norm)

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
            
            ax_norm.clear()
            ax_norm.plot(history["step"], history["w1_norm"], label="‖W1‖")
            ax_norm.plot(history["step"], history["w2_norm"], label="‖W2‖")
            ax_norm.plot(history["step"], history["total_norm"], label="total", linestyle="--")
            ax_norm.set_xscale("log")
            ax_norm.set_xlabel("step"); ax_norm.set_ylabel("Frobenius norm")
            ax_norm.set_title("Weight norms")
            ax_norm.legend(); ax_norm.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.01)
            
    # --- save model ---
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # weights only — small, portable, recommended
    torch.save(model.state_dict(), os.path.join(save_dir, "grok_model.pt"))

    # full checkpoint with training state — useful for analysis or resuming
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": num_steps,
        "history": history,
        "config": {
            "prime": prime,
            "d_model": 128,
            "num_heads": 4,
            "d_mlp": 512,
            "n_ctx": 3,
        },
    }, os.path.join(save_dir, "grok_checkpoint.pt"))

    print(f"Saved model to {save_dir}/")

    plt.ioff()
    plt.show()


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
    log_every = 100

    # Only track norms of weight matrices (2D+ tensors), not biases/LN scales.
    param_names = [name for name, p in model.named_parameters() if p.ndim >= 2]

    history = {"step": [], "train_loss": [], "test_loss": [],
               "train_acc": [], "test_acc": [],
               "total_norm": [],
               "param_norms": {name: [] for name in param_names}}

    plt.ion()

    n_params = len(param_names)
    n_cols = 4
    n_param_rows = (n_params + n_cols - 1) // n_cols
    n_rows = 1 + n_param_rows

    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    # Top row: loss, accuracy, total norm (spans the remaining columns).
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_total = fig.add_subplot(gs[0, 2:])

    param_axes = {}
    for i, name in enumerate(param_names):
        row = 1 + i // n_cols
        col = i % n_cols
        param_axes[name] = fig.add_subplot(gs[row, col])

    for step in range(num_steps):
        model.train()
        idx = torch.arange(0, len(train_inputs), device=device)
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

                per_param = {name: p.norm().item() for name, p in model.named_parameters()
                             if p.ndim >= 2}
                total_norm = sum(n**2 for n in per_param.values())**0.5

            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["total_norm"].append(total_norm)
            for name, n in per_param.items():
                history["param_norms"][name].append(n)

            print(f"Step {step:6d} | "
                  f"Train loss {loss.item():.4f} acc {train_acc:.3f} | "
                  f"Test loss {test_loss:.4f} acc {test_acc:.3f} | "
                  f"‖W‖ {total_norm:.2f}")

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

            ax_total.clear()
            ax_total.plot(history["step"], history["total_norm"], color="black")
            ax_total.set_xscale("log")
            ax_total.set_xlabel("step"); ax_total.set_ylabel("Frobenius norm")
            ax_total.set_title("Total ‖W‖")
            ax_total.grid(True, alpha=0.3)

            for name, ax in param_axes.items():
                ax.clear()
                ax.plot(history["step"], history["param_norms"][name])
                ax.set_xscale("log")
                ax.set_xlabel("step")
                ax.set_ylabel("‖·‖")
                ax.set_title(name, fontsize=9)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.pause(0.01)

    # --- save model ---
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # weights only — small, portable, recommended
    torch.save(model.state_dict(), os.path.join(save_dir, "grok_model.pt"))

    # full checkpoint with training state — useful for analysis or resuming
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": num_steps,
        "history": history,
        "config": {
            "prime": prime,
            "d_model": 128,
            "num_heads": 4,
            "d_mlp": 512,
            "n_ctx": 3,
        },
    }, os.path.join(save_dir, "grok_checkpoint.pt"))

    print(f"Saved model to {save_dir}/")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train_model()