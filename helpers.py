import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import GridSearchCV

from relio import Concept, CostFunction, DriftStatus, DriftType, OTMetric, Relio


def configure_page(page_title, heading, description, image=None, section_heading="Simulation:"):
    """Configure the Streamlit page and display its introduction."""
    st.set_page_config(
        page_title=page_title,
        page_icon="images/icon.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.logo("images/logo.png")
    st.title(heading)

    with st.expander(":blue[:question: What is concept drift?]", expanded=False):
        st.write(description)
        if image is not None:
            st.image(image)

    st.write(f"### {section_heading}")


def configure_parameters(defaults, include_model_type=True):
    metric_options = {
        "Wasserstein order 1": OTMetric.WASSERSTEIN1,
        "Wasserstein order 2": OTMetric.WASSERSTEIN2,
        "Regularized Wasserstein": OTMetric.SINKHORN,
    }

    cost_options = {
        "Euclidean": CostFunction.EUCLIDEAN,
        "Standardized Euclidean": CostFunction.SEUCLIDEAN,
        "Mahalanobis": CostFunction.MAHALANOBIS,
    }

    model_type = None

    with st.popover(":gear: Modify parameters"):
        st.write(":gear: Modify simulation parameters")

        if include_model_type:
            model_type = st.selectbox(
                "Choose the model type",
                [
                    "Supervised - Stochastic Gradient Descent",
                    "Unsupervised - KMeans",
                ],
            )

        window_size = st.number_input(
            "Enter the window size",
            min_value=1,
            value=defaults["window_size"],
        )

        metric_input = st.selectbox(
            "Choose the detection metric",
            list(metric_options),
            index=list(metric_options).index(defaults["metric"]),
        )

        cost_input = st.selectbox(
            "Choose the cost function",
            list(cost_options),
            index=list(cost_options).index(defaults["cost_function"]),
        )

        warning_threshold = st.number_input(
            "Enter the alert percentage",
            value=float(defaults["warning_threshold"]),
        )

        detection_threshold = st.number_input(
            "Enter the detection percentage",
            value=float(defaults["detection_threshold"]),
        )

        stability_threshold = st.number_input(
            "Enter the stability threshold",
            min_value=1,
            value=defaults["stability_threshold"],
        )

        sudden_drift_threshold = None

        if "sudden_drift_threshold" in defaults:
            sudden_drift_threshold = st.number_input(
                "Enter the sudden drift threshold",
                min_value=1,
                value=defaults["sudden_drift_threshold"],
                step=1,
            )

    return {
        "model_type": model_type,
        "window_size": window_size,
        "metric_input": metric_input,
        "ot_metric": metric_options[metric_input],
        "cost_input": cost_input,
        "cost_function": cost_options[cost_input],
        "warning_threshold": warning_threshold,
        "detection_threshold": detection_threshold,
        "stability_threshold": stability_threshold,
        "sudden_drift_threshold": sudden_drift_threshold,
    }


def prepare_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    pc1 = pca.fit_transform(df.iloc[:, :-1])

    return pca, pc1


def initialize_simulation(dataset_path, parameters, data_slice=None):
    df = pd.read_csv(dataset_path)

    if data_slice is not None:
        start, end = data_slice
        df = df.iloc[start:end].reset_index(drop=True)

    # Encode non-numerical labels
    labels = df.iloc[:, -1]

    try:
        encoded_labels = pd.to_numeric(labels).astype(int)
    except (ValueError, TypeError):
        encoded_labels, _ = pd.factorize(labels, sort=False)

        if -1 in encoded_labels:
            raise ValueError("The label column contains missing values.")

    df[df.columns[-1]] = pd.Series(
        encoded_labels, index=df.index, dtype="int64"
    )

    window_size = parameters["window_size"]

    api_parameters = {
        "window_size": window_size,
        "warning_threshold": parameters["warning_threshold"],
        "detection_threshold": parameters["detection_threshold"],
        "ot_metric": parameters["ot_metric"],
        "cost_function": parameters["cost_function"],
        "stability_threshold": parameters["stability_threshold"],
        "df": df,
    }

    if parameters["sudden_drift_threshold"] is not None:
        api_parameters["sudden_drift_threshold"] = parameters["sudden_drift_threshold"]

    relio = Relio(**api_parameters)

    ref_dist = df.iloc[:window_size].to_numpy()

    ref_dist_X = ref_dist[:, :-1]
    ref_dist_y = ref_dist[:, -1].astype(int)

    first_concept = Concept(1, ref_dist_X)

    relio.add_concept(first_concept)
    relio.set_curr_concept(first_concept)

    return {
        "df": df,
        "relio": relio,
        "ref_dist_X": ref_dist_X,
        "ref_dist_y": ref_dist_y,
        "all_classes": np.unique(df.iloc[:, -1].to_numpy()),
        "current_window": [],
        "drift_impacts": [],
        "adapt_perform": [],
    }


def display_configuration(parameters, include_model_type=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f":small_red_triangle_down: Window size: "
            f"***{parameters['window_size']} data points***",
            help=(
                "The number of data points considered for calculating the drift metric."
            ),
        )

        st.markdown(
            f":small_red_triangle_down: Detection metric: "
            f"***{parameters['metric_input']}***",
            help=(
                "The optimal transport metric used to compare data "
                "distributions and detect drift."
            ),
        )

        st.markdown(
            f":small_red_triangle_down: Cost function: "
            f"***{parameters['cost_input']}***",
            help=(
                "The distance computed between data points from two "
                "distributions, used by the optimal transport metric."
            ),
        )

        if include_model_type:
            st.markdown(
                f":small_red_triangle_down: Model type: ***{parameters['model_type']}***",
                help="Specifies whether the model is supervised or unsupervised.",
            )

    with col2:
        st.markdown(
            f":small_red_triangle_down: Alert percentage: "
            f"***{parameters['warning_threshold']}%***",
            help=(
                "The distribution shift percentage above which an alert is triggered."
            ),
        )

        st.markdown(
            f":small_red_triangle_down: Detection percentage: "
            f"***{parameters['detection_threshold']}%***",
            help=("The distribution shift percentage above which drift is detected."),
        )

        st.markdown(
            f":small_red_triangle_down: Stability threshold: "
            f"***{parameters['stability_threshold']} windows***",
            help=(
                "The number of windows used to determine whether the "
                "new distribution has remained stable."
            ),
        )

        if parameters["sudden_drift_threshold"] is not None:
            st.markdown(
                f":small_red_triangle_down: Sudden drift threshold: "
                f"***{parameters['sudden_drift_threshold']} windows***",
                help=(
                    "The maximum number of windows allowed for a transition "
                    "to be classified as sudden drift."
                ),
            )


def initialize_models(model_type, ref_dist_X, ref_dist_y):
    if model_type == "Supervised - Stochastic Gradient Descent":
        param_grid = {
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "penalty": ["l2", "l1", "elasticnet"],
            "max_iter": [1000, 2000, 3000],
        }

        grid_search = GridSearchCV(
            estimator=SGDClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            error_score="raise",
        )
        grid_search.fit(ref_dist_X, ref_dist_y)

        best_params = grid_search.best_params_

        model = SGDClassifier(**best_params, random_state=42)
        model.fit(ref_dist_X, ref_dist_y)

        drifted_model = SGDClassifier(**best_params, random_state=42)
        drifted_model.fit(ref_dist_X, ref_dist_y)

        metric_name = "Accuracy"

    elif model_type == "Unsupervised - KMeans":
        silhouette_scores = []

        for k in range(2, 11):
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(ref_dist_X)
            silhouette_scores.append(silhouette_score(ref_dist_X, cluster_labels))

        n_clusters = np.argmax(silhouette_scores) + 2

        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        drifted_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)

        model.fit(ref_dist_X)
        drifted_model.fit(ref_dist_X)

        cluster_labels_model = model.labels_
        cluster_labels_drift = drifted_model.labels_

        adapt_perform = [silhouette_score(ref_dist_X, cluster_labels_model)]
        drift_impacts = [silhouette_score(ref_dist_X, cluster_labels_drift)]

        metric_name = "Silhouette Score"

    return {
        "model": model,
        "drifted_model": drifted_model,
        "metric_name": metric_name,
        "adapt_perform": adapt_perform if model_type == "Unsupervised - KMeans" else [],
        "drift_impacts": drift_impacts if model_type == "Unsupervised - KMeans" else [],
    }


def initialize_simulation_display(pca, metric_name, metric_input):
    st.toast("Initializing simulation.", icon="⏳")

    st.write("##### :bar_chart: Evolution of the data distribution:")
    chart = st.empty()

    st.write(
        f"Quality of representation on axis 1 = "
        f"**{pca.explained_variance_ratio_[0] * 100:.2f}%**"
    )

    st.write(
        f"##### :chart_with_upwards_trend: Evolution of the "
        f"{metric_input} distance between the reference distribution "
        f"and the current window:"
    )
    distances = st.empty()

    st.divider()
    st.write(f"##### :chart_with_downwards_trend: Impact of drift - {metric_name}:")
    metric_chart = st.empty()

    st.divider()
    st.write("### :clock1: History of detected drifts:")

    return chart, distances, metric_chart


def display_drift_type(drift_type, detection_window, displayed_drifts):
    """Display a drift classification only once per detection window."""
    if drift_type is None or detection_window in displayed_drifts:
        return

    drift_names = {
        DriftType.GRADUAL: "Gradual",
        DriftType.SUDDEN: "Sudden",
        DriftType.RECURRENT: "Recurrent",
        DriftType.INCREMENTAL: "Incremental",
    }

    name = drift_names.get(drift_type)

    if name is None:
        return

    message = f"The drift type is: {name}"
    st.toast(f":blue[{message}]", icon="📌")
    st.info(message, icon="📌")

    displayed_drifts.add(detection_window)


def handle_drift_action(
    action,
    model_type,
    model,
    win_X,
    win_y,
    ref_dist_X,
    ref_dist_y,
    window_index,
):
    """Apply model adaptation or adjustment based on RELIO's action."""

    if action == DriftStatus.DRIFT:
        drift_time = datetime.datetime.now().strftime("%H:%M:%S")
        message = f"A drift is detected at data index {window_index} at {drift_time}."

        st.toast(message, icon="⚠️")
        st.error(message, icon="⚠️")

        train_X = np.concatenate((ref_dist_X, win_X))
        train_y = np.concatenate((ref_dist_y, win_y))

        if model_type == "Supervised - Stochastic Gradient Descent":
            model.fit(train_X, train_y)
        else:
            model.fit(train_X)

        ref_dist_X = win_X
        ref_dist_y = win_y

    elif action == DriftStatus.WARNING:
        alert_time = datetime.datetime.now().strftime("%H:%M:%S")
        message = f"A minor change in distribution occurred at {alert_time}."

        st.toast(message, icon="❗")
        st.warning(message, icon="❗")

    return ref_dist_X, ref_dist_y


def evaluate_window(model_type, model, drifted_model, win_X, win_y):
    """Evaluate adapted and non-adapted models on the current window."""

    if model_type == "Supervised - Stochastic Gradient Descent":
        metric = accuracy_score(win_y, model.predict(win_X))
        drifted_metric = accuracy_score(win_y, drifted_model.predict(win_X))

    else:
        labels = model.predict(win_X)
        drifted_labels = drifted_model.predict(win_X)

        metric = calculate_silhouette_score(win_X, labels)
        drifted_metric = calculate_silhouette_score(win_X, drifted_labels)

    return metric, drifted_metric


def calculate_silhouette_score(X, labels):
    """Calculate silhouette score when the clustering is valid."""

    n_clusters = len(np.unique(labels))

    if n_clusters < 2 or n_clusters >= len(X):
        return np.nan

    return silhouette_score(X, labels)


def update_simulation_charts(
    relio,
    distances,
    metric_chart,
    adapt_perform,
    drift_impacts,
):
    distances_data = pd.DataFrame({"distance": relio.get_distance_history()})
    distances_data["alert"] = relio.get_warning_thold()
    distances_data["detection"] = relio.get_detect_thold()

    distances.line_chart(
        distances_data,
        color=["#338AFF", "#FFAC1C", "#FF0D0D"],
    )

    metric_data = pd.DataFrame(
        {
            "with adaptation": adapt_perform,
            "without adaptation": drift_impacts,
        }
    )

    metric_chart.line_chart(
        metric_data,
        color=["#338AFF", "#FF0D0D"],
    )
