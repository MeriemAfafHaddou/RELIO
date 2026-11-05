import numpy as np
import pandas as pd
import streamlit as st

import relio
from helpers import (
    configure_page,
    configure_parameters,
    display_configuration,
    display_drift_type,
)

# Page configuration.
configure_page(
    page_title="Application to Social Media",
    heading="Drift Detection in Social Media Networks",
    description=(
        "This simulation evaluates RELIO's ability to detect distribution "
        "changes in evolving social media networks."
    ),
)

# Dataset configuration.
datasets = {
    "Cannes 2013": {
        "paths": [
            f"data/graphes/Cannes/Cannes2013_snapshot_{i}.edgelist.txt"
            for i in range(8)
        ],
        "warning_threshold": 0.43,
        "detection_threshold": 0.46,
    },
    "Digg": {
        "paths": [
            f"data/graphes/digg/Digg_snapshot_{i}.edgelist.txt"
            for i in range(1, 10)
        ],
        "warning_threshold": 0.50,
        "detection_threshold": 0.70,
    },
    "Cell": {
        "paths": [
            f"data/graphes/cell/real.t{i:02d}.edges"
            for i in range(1, 10)
        ] + ["data/graphes/cell/real.t010.edges"],
        "warning_threshold": 0.15,
        "detection_threshold": 0.17,
    },
}

# Dataset selection.
dataset_option = st.selectbox(
    "Which dataset would you like to pick?",
    tuple(datasets.keys()),
)

dataset_config = datasets[dataset_option]

df = [
    pd.read_csv(
        path,
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    for path in dataset_config["paths"]
]

# Simulation parameters.
defaults = {
    "window_size": 24,
    "metric": "Wasserstein order 2",
    "cost_function": "Standardized Euclidean",
    "warning_threshold": dataset_config["warning_threshold"],
    "detection_threshold": dataset_config["detection_threshold"],
    "stability_threshold": 3,
    "sudden_drift_threshold": 1,
}

parameters = configure_parameters(defaults)
display_configuration(parameters)

# Start button.
_, button_col = st.columns(2)

with button_col:
    start_button = st.button(":arrow_forward: Start", type="primary")

if start_button:

    # Initialize RELIO.
    api = relio.Relio(
        window_size=parameters["window_size"],
        warning_threshold=parameters["warning_threshold"],
        detection_threshold=parameters["detection_threshold"],
        ot_metric=parameters["ot_metric"],
        cost_function=parameters["cost_function"],
        stability_threshold=parameters["stability_threshold"],
        df=df,
        sudden_drift_threshold=parameters["sudden_drift_threshold"],
        time_mode=relio.TimeMode.PHYSICAL,
    )

    # Initialize the reference concept.
    ref_dist = df[0]
    first_concept = relio.Concept(1, np.array(ref_dist))

    api.add_concept(first_concept)
    api.set_curr_concept(first_concept)

    # Initialize display.
    st.write(
        f"""
        ##### :chart_with_upwards_trend: Evolution of the
        {parameters["metric_input"]} distance:
        """
    )

    distances = st.empty()

    st.divider()

    st.write("### :clock1: History of detected drifts")

    displayed_drifts = set()

    # Process graph snapshots.
    for i, current_window in enumerate(df[1:], start=1):

        api.set_curr_win(np.array(current_window))
        api.detect_drift()

        action = api.get_action()

        if action == relio.DriftStatus.DRIFT:

            drift_time = pd.Timestamp.now().strftime("%H:%M:%S")

            st.toast(
                f":red[Drift detected at snapshot {i + 1} "
                f"({drift_time})]",
                icon="⚠️",
            )

            st.error(
                f"Drift detected at snapshot {i + 1} ({drift_time})",
                icon="⚠️",
            )

            api.reset_adapt_model()

            ref_dist = current_window

        elif action == relio.DriftStatus.WARNING:

            alert_time = pd.Timestamp.now().strftime("%H:%M:%S")

            st.toast(
                f"Alert: A minor distribution change occurred at "
                f"snapshot {i + 1} ({alert_time})!",
                icon="❗",
            )

            st.warning(
                f"Alert: A minor distribution change occurred at "
                f"snapshot {i + 1} ({alert_time})!",
                icon="❗",
            )

            api.reset_adjust_model()

        # Update distance chart.
        distances_data = pd.DataFrame(
            api.get_distance_history(),
            columns=["distance"],
        )

        distances_data["alert"] = api.get_warning_thold()
        distances_data["detection"] = api.get_detect_thold()

        distances.line_chart(
            distances_data,
            color=["#338AFF", "#FFAC1C", "#FF0D0D"],
        )

        # Display drift classification.
        drift_type = api.identify_drift_type()

        display_drift_type(
            drift_type,
            i + 1,
            displayed_drifts,
        )
