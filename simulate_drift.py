import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def load_iris_classes(csv_path):
    """
    Load the Iris dataset from a CSV file.
    """
    df = pd.read_csv(csv_path)

    class_0 = df[:50]
    class_1 = df[50:100]
    class_2 = df[100:]

    return class_0, class_1, class_2


def generate_sudden(old_concept, new_concept, window_size=20):
    """
    Generate a dataset containing sudden concept drift.

    The stream starts with the old concept, followed by the new concept.
    All complete windows are used, and the transition occurs at a
    window boundary without intermediate distributions.
    """

    # Shuffle observations while preserving their original distributions.
    old_samples = old_concept.sample(frac=1, random_state=42)
    new_samples = new_concept.sample(frac=1, random_state=43)

    # Keep only complete windows.
    n_observations = (len(new_samples) // window_size) * window_size
    new_samples = new_samples.iloc[:n_observations]

    # Concatenate concepts in temporal order.
    dataset = pd.concat([old_samples, new_samples], ignore_index=True)

    # Remove labels because drift detection uses feature distributions.
    dataset = dataset.drop(columns=["species"])

    return dataset


def generate_recurrent(old_concept, new_concept, window_size=20):
    """
    Generate a dataset containing recurrent concept drift.

    The stream follows the sequence:
        - 3 windows from the old concept.
        - 3 windows from the new concept.
        - 2 windows from the old concept.
        - 4 windows from the new concept.
    """

    # Shuffle each concept independently.
    old_samples = old_concept.sample(frac=1, random_state=42)
    new_samples = new_concept.sample(frac=1, random_state=43)

    # Divide observations into recurring segments.
    segments = [
        old_samples.iloc[: 3 * window_size],
        new_samples.iloc[: 3 * window_size],
        old_samples.iloc[3 * window_size : 5 * window_size],
        new_samples.iloc[3 * window_size : 7 * window_size],
    ]

    # Concatenate segments in temporal order.
    dataset = pd.concat(segments, ignore_index=True)

    # Remove labels because drift detection uses feature distributions.
    dataset = dataset.drop(columns=["species"])

    return dataset


def generate_gradual(old_concept, new_concept, window_size=20):
    """
    Generate a dataset containing gradual concept drift.

    Each intermediate window contains observations from both concepts,
    with the proportion of the new concept increasing over time.
    """

    # Shuffle observations from each concept.
    old_samples = old_concept.sample(frac=1, random_state=42)
    new_samples = new_concept.sample(frac=1, random_state=43)

    # Define the number of observations from each concept per window.
    old_counts = [20, 17, 14, 10, 6, 3, 0]

    windows = []
    old_index = 0
    new_index = 0

    for old_count in old_counts:
        new_count = window_size - old_count

        old_window = old_samples.iloc[old_index : old_index + old_count]
        new_window = new_samples.iloc[new_index : new_index + new_count]

        old_index += old_count
        new_index += new_count

        window = pd.concat([old_window, new_window])
        windows.append(window)

    # Concatenate windows in temporal order.
    dataset = pd.concat(windows, ignore_index=True)

    # Remove labels because drift detection uses feature distributions.
    dataset = dataset.drop(columns=["species"])

    return dataset


def generate_incremental(old_concept, new_concept, window_size=20):
    """
    Generate a dataset containing incremental concept drift.

    The stream contains:
        - 3 windows from the old concept.
        - 3 intermediate windows generated through interpolation.
        - 3 windows from the new concept.
    """

    # Shuffle observations from both concepts.
    old_samples = old_concept.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    new_samples = new_concept.sample(
        frac=1, random_state=43
    ).reset_index(drop=True)

    # Reserve observations for the old and new concept windows.
    old_windows = old_samples.iloc[:3 * window_size]
    new_windows = new_samples.iloc[:3 * window_size]

    # Select observations for generating intermediate distributions.
    # The old selection overlaps with the old windows by 20 observations.
    old_values = old_samples.iloc[2 * window_size:5 * window_size]
    new_values = new_samples.iloc[3 * window_size:6 * window_size]

    # Remove labels before matching and interpolation.
    old_values = old_values.drop(columns=["species"]).to_numpy()
    new_values = new_values.drop(columns=["species"]).to_numpy()

    # Standardize both populations using a common feature scale.
    scaler = StandardScaler()
    combined = np.vstack([old_values, new_values])
    scaler.fit(combined)

    old_scaled = scaler.transform(old_values)
    new_scaled = scaler.transform(new_values)

    # Find the one-to-one assignment minimizing total distance.
    distances = cdist(old_scaled, new_scaled, metric="euclidean")
    old_indices, new_indices = linear_sum_assignment(distances)

    old_values = old_values[old_indices]
    new_values = new_values[new_indices]

    # Shuffle matched pairs while preserving their correspondence.
    permutation = np.random.default_rng(44).permutation(len(old_values))
    old_values = old_values[permutation]
    new_values = new_values[permutation]

    # Generate three intermediate windows.
    windows = [old_windows.drop(columns=["species"])]

    for i, lam in enumerate([0.25, 0.50, 0.75]):
        start = i * window_size
        end = start + window_size

        interpolated = (
            (1 - lam) * old_values[start:end]
            + lam * new_values[start:end]
        )

        windows.append(
            pd.DataFrame(
                interpolated,
                columns=old_concept.drop(columns=["species"]).columns
            )
        )

    windows.append(new_windows.drop(columns=["species"]))

    # Concatenate windows in temporal order.
    return pd.concat(windows, ignore_index=True)

def main():
    # Load dataset
    class_0, class_1, class_2 = load_iris_classes("data/iris-classes.csv")

    # Define concepts
    old_concept = pd.concat([class_1, class_2])
    new_concept = pd.concat([class_0, class_1, class_2])
    sudden_dataset = generate_sudden(old_concept, new_concept)
    recurrent_dataset = generate_recurrent(old_concept, new_concept)
    gradual_dataset = generate_gradual(old_concept, new_concept)
    incremental_dataset = generate_incremental(old_concept, new_concept)

    sudden_dataset.to_csv("data/iris_sudden.csv", index=False)
    recurrent_dataset.to_csv("data/iris_recurrent.csv", index=False)
    gradual_dataset.to_csv("data/iris_gradual.csv", index=False)
    incremental_dataset.to_csv("data/iris_incremental.csv", index=False)

if __name__ == "__main__":
    main()
