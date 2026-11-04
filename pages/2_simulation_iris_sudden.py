import numpy as np
import streamlit as st

from helpers import (
    configure_page,
    configure_parameters,
    display_configuration,
    display_drift_type,
    evaluate_window,
    handle_drift_action,
    initialize_models,
    initialize_simulation,
    initialize_simulation_display,
    prepare_pca,
    update_simulation_charts,
)
from relio import DriftStatus

# Page configuration.
configure_page(
    page_title="Sudden drift simulation",
    heading="Sudden drift simulation",
    description=(
        "Sudden concept drift occurs when the model built on a certain "
        "concept becomes obsolete because the data distribution changes "
        "abruptly, as shown in the figure:"
    ),
    image="images/soudain.png",
)

# Default parameter values.
defaults = {
    "window_size": 20,
    "metric": "Wasserstein order 2",
    "cost_function": "Standardized Euclidean",
    "warning_threshold": 15,
    "detection_threshold": 30,
    "stability_threshold": 3,
    "sudden_drift_threshold": 2,
}

# Configure simulation parameters.
parameters = configure_parameters(defaults)
display_configuration(parameters)

# Start button.
_, button_col = st.columns(2)

with button_col:
    start_button = st.button(":arrow_forward: Start", type="primary")


if start_button:

    # Initialize the dataset and RELIO API.
    simulation = initialize_simulation(
        dataset_path="data/iris_sudden.csv",
        parameters=parameters,
    )

    df = simulation["df"]
    relio = simulation["relio"]
    ref_dist_X = simulation["ref_dist_X"]
    ref_dist_y = simulation["ref_dist_y"]
    current_window = simulation["current_window"]

    # Initialize the models.
    models = initialize_models(
        parameters["model_type"],
        ref_dist_X,
        ref_dist_y,
    )

    model = models["model"]
    drifted_model = models["drifted_model"]
    metric_name = models["metric_name"]
    adapt_perform = models["adapt_perform"]
    drift_impacts = models["drift_impacts"]

    # Prepare the data for visualization.
    pca, pc1 = prepare_pca(df)

    # Initialize the charts.
    chart, distances, metric_chart = initialize_simulation_display(
        pca,
        metric_name,
        parameters["metric_input"],
    )

    # Simulation parameters.
    window_size = parameters["window_size"]
    model_type = parameters["model_type"]
    displayed_drifts = set()

    # Process the data stream.
    for i in range(window_size, len(df) + 1):

        chart.line_chart(pc1[:i])

        current_window.append(df.iloc[i - 1])

        if len(current_window) == window_size:

            win = np.array(current_window)
            win_X = win[:, :-1]
            win_y = win[:, -1].astype(int)

            # Detect drift.
            relio.set_curr_win(win_X)
            drift_type = relio.detect_drift()
            action = relio.get_action()

            # Handle model adaptation or warning.
            ref_dist_X, ref_dist_y = handle_drift_action(
                action,
                model_type,
                model,
                win_X,
                win_y,
                ref_dist_X,
                ref_dist_y,
                i,
            )

            if action == DriftStatus.DRIFT:
                relio.reset_adapt_model()

            elif action == DriftStatus.WARNING:
                relio.reset_adjust_model()

            # Display drift classification when available.
            display_drift_type(drift_type, i, displayed_drifts)

            # Evaluate model performance.
            metric, drifted_metric = evaluate_window(
                model_type,
                model,
                drifted_model,
                win_X,
                win_y,
            )

            adapt_perform.append(metric)
            drift_impacts.append(drifted_metric)

            # Update the charts.
            update_simulation_charts(
                relio,
                distances,
                metric_chart,
                adapt_perform,
                drift_impacts,
            )

            current_window = []
