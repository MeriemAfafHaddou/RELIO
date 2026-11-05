import datetime

import numpy as np
import pandas as pd
import streamlit as st
from frouros.callbacks.batch import PermutationTestDistanceBased
from frouros.detectors.data_drift import EMD, JS
from sklearn.preprocessing import LabelEncoder

import relio
from helpers import (
    configure_page,
    configure_parameters,
    display_configuration,
    display_drift_type,
    prepare_pca,
)

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

configure_page(
    page_title="Comparaison",
    heading="Comparison between RELIO and existing methods",
    description=(
        "Concept drift occurs when the statistical properties of data "
        "change over time. This page compares RELIO with existing "
        "distribution-based drift detection methods."
    ),
    section_heading="Comparison:",
)


# ------------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------------

DATASET_CONFIG = {
    "Asfault": {
        "path": "data/Asfault.csv",
        "header": None,
        "data_slice": (0, 8000),
        "label_column": 64,
        "window_size": 500,
        "metric": "Wasserstein order 2",
        "cost_function": "Standardized Euclidean",
        "warning_threshold": 10,
        "detection_threshold": 20,
        "stability_threshold": 3,
    },
    "Ozone": {
        "path": "data/Ozone.csv",
        "header": None,
        "label_column": 72,
        "window_size": 150,
        "metric": "Wasserstein order 2",
        "cost_function": "Standardized Euclidean",
        "warning_threshold": 50,
        "detection_threshold": 70,
        "stability_threshold": 3,
    },
    "Simulation : Iris Sudden": {
        "path": "data/iris_sudden.csv",
        "window_size": 20,
        "metric": "Wasserstein order 2",
        "cost_function": "Standardized Euclidean",
        "warning_threshold": 15,
        "detection_threshold": 30,
        "stability_threshold": 3,
        "sudden_drift_threshold": 2,
    },
    "Simulation : Iris Gradual": {
        "path": "data/iris_gradual.csv",
        "window_size": 20,
        "metric": "Wasserstein order 2",
        "cost_function": "Standardized Euclidean",
        "warning_threshold": 5,
        "detection_threshold": 10,
        "stability_threshold": 3,
    },
    "Simulation : Iris Recurrent": {
        "path": "data/iris_recurrent.csv",
        "window_size": 20,
        "metric": "Wasserstein order 2",
        "cost_function": "Standardized Euclidean",
        "warning_threshold": 15,
        "detection_threshold": 25,
        "stability_threshold": 4,
    },
    "Simulation : Iris Incremental": {
        "path": "data/iris_incremental.csv",
        "window_size": 40,
        "metric": "Wasserstein order 2",
        "cost_function": "Standardized Euclidean",
        "warning_threshold": 10,
        "detection_threshold": 15,
        "stability_threshold": 3,
    },
    "Synthetic : Insects Sudden": {
        "path": "data/insects_sudden.csv",
        "header": None,
        "data_slice": (9800, 13800),
        "window_size": 200,
        "metric": "Wasserstein order 2",
        "cost_function": "Standardized Euclidean",
        "warning_threshold": 40,
        "detection_threshold": 60,
        "stability_threshold": 3,
    },
    "Synthetic : Insects Incremental": {
        "path": "data/insects_incremental.csv",
        "header": None,
        "data_slice": (32000, 37000),
        "window_size": 200,
        "metric": "Wasserstein order 2",
        "cost_function": "Standardized Euclidean",
        "warning_threshold": 20,
        "detection_threshold": 40,
        "stability_threshold": 3,
    },
}

option = st.selectbox(
    ":bar_chart: Which dataset would you like to pick?",
    list(DATASET_CONFIG.keys()),
)


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------


dataset_config = DATASET_CONFIG[option]

df = pd.read_csv(
    dataset_config["path"],
    header=dataset_config.get("header", "infer"),
)

# Apply the selected data slice.
if "data_slice" in dataset_config:
    start, end = dataset_config["data_slice"]
    df = df.iloc[start:end].copy()

# Encode the class column when its position is explicitly defined.
if "label_column" in dataset_config:
    label_column = dataset_config["label_column"]

    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df[label_column])

    df = df.drop(columns=[label_column])

else:
    # Encode the existing target column.
    labels = df.iloc[:, -1]

    try:
        encoded_labels = pd.to_numeric(labels).astype(int)
    except (ValueError, TypeError):
        encoded_labels, _ = pd.factorize(labels, sort=False)

        if -1 in encoded_labels:
            raise ValueError("The label column contains missing values.")

    df.iloc[:, -1] = encoded_labels

df = df.reset_index(drop=True)

defaults = {
    "window_size": dataset_config["window_size"],
    "warning_threshold": dataset_config["warning_threshold"],
    "detection_threshold": dataset_config["detection_threshold"],
    "metric": "Wasserstein order 2",
    "cost_function": "Standardized Euclidean",
    "stability_threshold": 4,
}


# ------------------------------------------------------------------
# Parameter configuration
# ------------------------------------------------------------------

parameters = configure_parameters(
    defaults,
    include_model_type=False,
)

display_configuration(
    parameters,
    include_model_type=False,
)


# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

window_size = parameters["window_size"]

if len(df) < 2 * window_size:
    st.error(
        "The selected dataset must contain at least two complete windows."
    )
    st.stop()

# PCA representation for Frouros.
pca, pc1 = prepare_pca(df, n_components=1)

# RELIO initialization.
api = relio.Relio(
    window_size=window_size,
    warning_threshold=parameters["warning_threshold"],
    detection_threshold=parameters["detection_threshold"],
    ot_metric=parameters["ot_metric"],
    cost_function=parameters["cost_function"],
    stability_threshold=parameters["stability_threshold"],
    df=df,
)

# Initial reference distribution.
ref_dist = df.iloc[:window_size].to_numpy()

first_concept = relio.Concept(
    1,
    ref_dist[:, :-1],
)

api.add_concept(first_concept)
api.set_curr_concept(first_concept)

ref_dist_X = ref_dist[:, :-1]
ref_dist_y = ref_dist[:, -1].astype(int)

# Frouros reference distribution.
fr_ref = pc1[:window_size]


# ------------------------------------------------------------------
# Frouros detectors
# ------------------------------------------------------------------

def initialize_frouros_detector(detector_class):
    return detector_class(
        callbacks=[
            PermutationTestDistanceBased(
                num_permutations=1000,
                random_state=31,
                num_jobs=-1,
                method="exact",
                name="permutation_test",
                verbose=True,
            ),
        ],
    )


emd_detector = initialize_frouros_detector(EMD)
js_detector = initialize_frouros_detector(JS)

emd_detector.fit(X=fr_ref)
js_detector.fit(X=fr_ref)

ALPHA = 0.05


# ------------------------------------------------------------------
# Comparison interface
# ------------------------------------------------------------------

st.write("")

col1, col2 = st.columns(2)

with col1:
    st.write(
        f":small_red_triangle_down: Window size: "
        f"***{window_size} data points***"
    )

    st.write(
        f":small_red_triangle_down: Detection metric: "
        f"***{parameters['metric_input']}***"
    )

    st.write(
        f":small_red_triangle_down: Cost function: "
        f"***{parameters['cost_input']}***"
    )

with col2:
    st.write(
        f":small_red_triangle_down: Alert percentage: "
        f"***{parameters['warning_threshold']}%***"
    )

    st.write(
        f":small_red_triangle_down: Detection percentage: "
        f"***{parameters['detection_threshold']}%***"
    )

    st.write(
        f":small_red_triangle_down: Stability threshold: "
        f"***{parameters['stability_threshold']} windows***"
    )


# ------------------------------------------------------------------
# Run comparison
# ------------------------------------------------------------------

button = st.button(
    ":arrow_forward: Launch comparison",
    type="primary",
)

if button:

    st.toast("Initializing the comparison...", icon="⏳")

    st.write("##### :bar_chart: Evolution of the data distribution:")

    chart = st.empty()

    st.write(
        f"Quality of representation on axis 1 = "
        f"**{pca.explained_variance_ratio_[0] * 100:.2f}%**"
    )

    st.divider()

    st.write("### :scales: Detection results comparison")

    relio_col, fr_col1, fr_col2 = st.columns(3)

    with relio_col:
        st.markdown("##### RELIO")
        st.write("##### :chart_with_upwards_trend: RELIO metric evolution")
        distances_relio = st.empty()
        st.divider()

    with fr_col1:
        st.markdown(
            "##### Frouros - EMD",
            help="Earth Mover's Distance: 1-Wasserstein.",
        )
        st.write("##### :chart_with_upwards_trend: EMD distance evolution")
        distances_emd = st.empty()
        st.divider()

    with fr_col2:
        st.markdown(
            "##### Frouros - JS",
            help="Jensen-Shannon distance based on KL divergence.",
        )
        st.write("##### :chart_with_upwards_trend: JS distance evolution")
        distances_js = st.empty()
        st.divider()

    # Comparison histories.
    dist_emd = []
    dist_js = []

    current_window = []
    displayed_drifts = set()

    # Process observations after the initial reference window.
    for i in range(window_size, len(df)):

        chart.line_chart(pc1[:i + 1])

        current_window.append(df.iloc[i].to_numpy())

        if len(current_window) < window_size:
            continue

        # Current window.
        win = np.asarray(current_window)

        win_X = win[:, :-1]
        win_y = win[:, -1].astype(int)

        fr_win = pc1[i - window_size + 1:i + 1]

        # ----------------------------------------------------------
        # RELIO
        # ----------------------------------------------------------

        api.set_curr_win(win_X)
        api.detect_drift()

        action = api.get_action()

        if action == relio.DriftStatus.DRIFT:

            drift_time = datetime.datetime.now().strftime("%H:%M:%S")

            message = (
                f"RELIO: A drift is detected in data with index "
                f"{i + 1 - window_size} at {drift_time}."
            )

            st.toast(message, icon="⚠️")

            with relio_col:
                st.error(message, icon="⚠️")

            drift_type = api.identify_drift_type()

            with relio_col:
                display_drift_type(
                    drift_type,
                    detection_window=i,
                    displayed_drifts=displayed_drifts,
                )

            api.reset_adapt_model()

            ref_dist_X = win_X
            ref_dist_y = win_y

        elif action == relio.DriftStatus.WARNING:

            alert_time = datetime.datetime.now().strftime("%H:%M:%S")

            message = (
                f"RELIO: A minor change in distribution occurred "
                f"at data index {i + 1 - window_size} at {alert_time}."
            )

            st.toast(message, icon="❗")

            with relio_col:
                st.warning(message, icon="❗")

            api.reset_adjust_model()

        # RELIO distance history.
        distances_data = pd.DataFrame(
            {
                "distance": api.get_distance_history(),
                "alert": api.get_warning_thold(),
                "detection": api.get_detect_thold(),
            }
        )

        with relio_col:
            distances_relio.line_chart(
                distances_data,
                color=["#338AFF", "#FFAC1C", "#FF0D0D"],
            )

        # ----------------------------------------------------------
        # FROUROS - EMD
        # ----------------------------------------------------------

        emd_dist, callbacks_log = emd_detector.compare(X=fr_win)

        p_value = callbacks_log["permutation_test"]["p_value"]

        if p_value <= ALPHA:

            drift_time = datetime.datetime.now().strftime("%H:%M:%S")

            with fr_col1:
                st.error(
                    f"Frouros EMD: Drift detected at data index "
                    f"{i + 1 - window_size}, {drift_time}.",
                    icon="⚠️",
                )

            emd_detector.fit(X=fr_win)

        dist_emd.append(emd_dist)

        with fr_col1:
            distances_emd.line_chart(dist_emd, color=["#338AFF"])

        # ----------------------------------------------------------
        # FROUROS - JS
        # ----------------------------------------------------------

        js_dist, callbacks_log = js_detector.compare(X=fr_win)

        p_value = callbacks_log["permutation_test"]["p_value"]

        if p_value <= ALPHA:

            drift_time = datetime.datetime.now().strftime("%H:%M:%S")

            with fr_col2:
                st.error(
                    f"Frouros JS: Drift detected at data index "
                    f"{i + 1 - window_size}, {drift_time}.",
                    icon="⚠️",
                )

            js_detector.fit(X=fr_win)

        dist_js.append(js_dist)

        with fr_col2:
            distances_js.line_chart(dist_js, color=["#338AFF"])

        # Reset the current window.
        current_window = []
