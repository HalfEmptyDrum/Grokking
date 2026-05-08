import os
import csv
import random


def generate_dataset(prime: int = 113):
    dataset = []
    for a in range(prime):
        for b in range(prime):
            result = (a + b) % prime
            dataset.append((a, b, result))

    data_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(data_dir, "dataset.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "result"])
        writer.writerows(dataset)

    return dataset


def split_training_test(dataset=None, train_fraction=0.3, seed=42):
    if dataset is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(data_dir, "dataset.csv")
        dataset = []
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                dataset.append((int(row[0]), int(row[1]), int(row[2])))

    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split = int(len(dataset) * train_fraction)
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]

    data_dir = os.path.dirname(os.path.abspath(__file__))
    for name, data in [("train.csv", train_data), ("test.csv", test_data)]:
        filepath = os.path.join(data_dir, name)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "result"])
            writer.writerows(data)

    return train_data, test_data


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(data_dir, "dataset.csv")
    if os.path.exists(dataset_path):
        print("Dataset already exists, skipping generation.")
        dataset = None
    else:
        dataset = generate_dataset()
    train_data, test_data = split_training_test(dataset)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
