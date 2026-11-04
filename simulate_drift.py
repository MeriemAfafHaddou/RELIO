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


def generate_gradual(old_concept, new_concept, window_size=20, n_windows=15):
    """
    Generate a dataset containing gradual concept drift.

    Each intermediate window contains observations from both concepts.
    The proportion of the new concept increases progressively.
    """

    if n_windows < 3:
        raise ValueError("n_windows must be at least 3.")

    old_proportions = np.linspace(1.0, 0.0, n_windows)

    old_counts = [round(window_size * p) for p in old_proportions]
    new_counts = [window_size - count for count in old_counts]

    required_old = sum(old_counts)
    required_new = sum(new_counts)

    old_samples = old_concept.sample(
        n=required_old,
        replace=required_old > len(old_concept),
        random_state=42,
    ).reset_index(drop=True)

    new_samples = new_concept.sample(
        n=required_new,
        replace=required_new > len(new_concept),
        random_state=43,
    ).reset_index(drop=True)

    windows = []
    old_index = 0
    new_index = 0

    for i, (old_count, new_count) in enumerate(zip(old_counts, new_counts)):
        old_window = old_samples.iloc[old_index : old_index + old_count]

        new_window = new_samples.iloc[new_index : new_index + new_count]

        old_index += old_count
        new_index += new_count

        window = pd.concat([old_window, new_window])

        window = window.sample(frac=1, random_state=100 + i).reset_index(drop=True)

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


def generate_incremental(
    old_concept,
    new_concept,
    window_size=20,
    n_before=10,
    n_transition=10,
    n_after=10,
):
    """
    Generate a dataset containing incremental concept drift.

    The stream contains:
        - n_before windows from the old concept.
        - n_transition windows with progressively changing
          feature and class distributions.
        - n_after windows from the new concept.

    Species labels are preserved throughout the stream.
    """

    classes = np.union1d(
        old_concept["species"].unique(),
        new_concept["species"].unique(),
    )

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

    windows = []

    def sample_concept(concept, species, count, seed):
        if count == 0:
            return pd.DataFrame(columns=concept.columns)

        population = concept[concept["species"] == species]

        if population.empty:
            return pd.DataFrame(columns=concept.columns)

        return population.sample(
            n=count,
            replace=count > len(population),
            random_state=seed,
        )

    # Before drift: observations from the old concept.
    for i in range(n_before):
        window = old_concept.sample(
            n=window_size,
            replace=window_size > len(old_concept),
            random_state=100 + i,
        )
        windows.append(window)

    # During drift: progressively mix both concepts.
    for i in range(n_transition):

        lam = (i + 1) / (n_transition + 1)

        class_proportions = (1 - lam) * old_proportions + lam * new_proportions

        class_counts = allocate_class_counts(
            class_proportions,
            window_size,
        )

        class_samples = []

        for j, (species, count) in enumerate(zip(classes, class_counts)):

            old_count = round(count * (1 - lam))
            new_count = count - old_count

            old_samples = sample_concept(
                old_concept, species, old_count, 200 + i * 100 + j
            )

            new_samples = sample_concept(
                new_concept, species, new_count, 300 + i * 100 + j
            )

            class_samples.extend([old_samples, new_samples])

        window = pd.concat(class_samples, ignore_index=True)

        window = window.sample(
            frac=1,
            random_state=400 + i,
        ).reset_index(drop=True)

        windows.append(window)

    # After drift: observations from the new concept.
    for i in range(n_after):
        window = new_concept.sample(
            n=window_size,
            replace=window_size > len(new_concept),
            random_state=500 + i,
        )
        windows.append(window)

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
