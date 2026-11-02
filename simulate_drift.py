import numpy as np
import pandas as pd


def load_iris_classes(csv_path):
    """
    Load the Iris dataset and separate its three classes.
    """
    df = pd.read_csv(csv_path)

    # Encode species as numeric labels.
    df["species"] = df["species"].map(
        {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    )

    class_0 = df.iloc[:50].copy()
    class_1 = df.iloc[50:100].copy()
    class_2 = df.iloc[100:].copy()

    return class_0, class_1, class_2


def generate_sudden(old_concept, new_concept, window_size=20):
    """
    Generate a dataset containing sudden concept drift.

    The stream starts with the old concept, followed by the new concept.
    The transition occurs at a window boundary.
    """

    old_samples = old_concept.sample(frac=1, random_state=42)

    new_samples = new_concept.sample(frac=1, random_state=43)

    # Keep only complete windows from the new concept.
    n_observations = (len(new_samples) // window_size) * window_size
    new_samples = new_samples.iloc[:n_observations]

    # Concatenate concepts in temporal order.
    return pd.concat([old_samples, new_samples], ignore_index=True)


def generate_recurrent(old_concept, new_concept, window_size=20):
    """
    Generate a dataset containing recurrent concept drift.

    The stream follows the sequence:
        - 3 windows from the old concept.
        - 3 windows from the new concept.
        - 2 windows from the old concept.
        - 4 windows from the new concept.
    """

    old_samples = old_concept.sample(frac=1, random_state=42)

    new_samples = new_concept.sample(frac=1, random_state=43)

    segments = [
        old_samples.iloc[: 3 * window_size],
        new_samples.iloc[: 3 * window_size],
        old_samples.iloc[3 * window_size : 5 * window_size],
        new_samples.iloc[3 * window_size : 7 * window_size],
    ]

    return pd.concat(segments, ignore_index=True)


def generate_gradual(old_concept, new_concept, window_size=20):
    """
    Generate a dataset containing gradual concept drift.

    Each intermediate window contains observations from both concepts.
    The proportion of the new concept increases over time.
    """

    old_samples = old_concept.sample(frac=1, random_state=42)

    new_samples = new_concept.sample(frac=1, random_state=43)

    # Proportion of old-concept observations in each window.
    old_proportions = [1.0, 0.85, 0.70, 0.50, 0.30, 0.15, 0.0]

    windows = []
    old_index = 0
    new_index = 0

    for proportion in old_proportions:
        old_count = round(window_size * proportion)
        new_count = window_size - old_count

        old_window = old_samples.iloc[old_index : old_index + old_count]

        new_window = new_samples.iloc[new_index : new_index + new_count]

        old_index += old_count
        new_index += new_count

        window = pd.concat([old_window, new_window])
        windows.append(window)

    return pd.concat(windows, ignore_index=True)


def allocate_class_counts(class_proportions, window_size):
    """
    Convert class proportions into integer counts summing to window_size.
    Uses the largest-remainder method.
    """

    expected_counts = class_proportions * window_size
    counts = np.floor(expected_counts).astype(int)

    remaining = window_size - counts.sum()

    if remaining > 0:
        remainders = expected_counts - counts
        indices = np.argsort(-remainders, kind="stable")
        counts[indices[:remaining]] += 1

    return counts


def generate_incremental(old_concept, new_concept, window_size=20):
    """
    Generate a dataset containing incremental concept drift.

    The stream contains:
        - 3 windows from the old concept.
        - 3 intermediate windows with progressively changing
          class proportions.
        - 3 windows from the new concept.

    Species labels are preserved throughout the stream.
    """

    old_samples = old_concept.sample(frac=1, random_state=42).reset_index(drop=True)

    new_samples = new_concept.sample(frac=1, random_state=43).reset_index(drop=True)

    # Reserve observations for the initial and final concepts.
    old_windows = old_samples.iloc[: 3 * window_size]
    new_windows = new_samples.iloc[: 3 * window_size]

    # Determine the class distribution of each concept.
    classes = new_concept["species"].unique()

    old_proportions = (
        old_concept["species"]
        .value_counts(normalize=True)
        .reindex(classes, fill_value=0)
        .to_numpy()
    )

    new_proportions = (
        new_concept["species"]
        .value_counts(normalize=True)
        .reindex(classes, fill_value=0)
        .to_numpy()
    )

    # Generate three intermediate windows.
    windows = [old_windows]

    lambdas = [0.25, 0.50, 0.75]

    for i, lam in enumerate(lambdas):
        # Gradually transition from the old class distribution
        # to the new class distribution.
        class_proportions = (1 - lam) * old_proportions + lam * new_proportions

        class_counts = allocate_class_counts(class_proportions, window_size)

        class_samples = []

        for j, (species, count) in enumerate(zip(classes, class_counts)):
            if count == 0:
                continue

            population = new_concept[new_concept["species"] == species]

            samples = population.sample(
                n=count, replace=count > len(population), random_state=100 + i * 10 + j
            )

            class_samples.append(samples)

        window = pd.concat(class_samples)

        # Shuffle observations within the intermediate window.
        window = window.sample(frac=1, random_state=200 + i).reset_index(drop=True)

        windows.append(window)

    windows.append(new_windows)

    return pd.concat(windows, ignore_index=True)


def main():
    # Load dataset.
    class_0, class_1, class_2 = load_iris_classes("data/iris-classes.csv")

    # Define concepts.
    old_concept = pd.concat([class_1, class_2])
    new_concept = pd.concat([class_0, class_1, class_2])

    # Generate drift datasets.
    sudden_dataset = generate_sudden(old_concept, new_concept)

    recurrent_dataset = generate_recurrent(old_concept, new_concept)

    gradual_dataset = generate_gradual(old_concept, new_concept)

    incremental_dataset = generate_incremental(old_concept, new_concept)

    # Save datasets with their species labels.
    sudden_dataset.to_csv("data/iris_sudden.csv", index=False)

    recurrent_dataset.to_csv("data/iris_recurrent.csv", index=False)

    gradual_dataset.to_csv("data/iris_gradual.csv", index=False)

    incremental_dataset.to_csv("data/iris_incremental.csv", index=False)


if __name__ == "__main__":
    main()
