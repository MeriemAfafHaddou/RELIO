import datetime

import numpy as np
import pandas as pd
import streamlit as st
from frouros.datasets.synthetic import SEA
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import GridSearchCV

import relio_api as relio

st.logo("images/logo.png")
st.set_page_config(
    page_title="Generated Data",
    page_icon="images/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
pca = PCA(n_components=2)
st.write("""
# Tests on generated data by a Streaming Ensemble Algorithm (SEA) from Frouros
""")
st.write("""
         ### Test :
""")

st.write(
    " * We generated a dataset of **500 data points**, with a **rate of "
    "noisy-class samples equal to 0.4**"
)
sea = SEA(seed=31)
it = sea.generate_dataset(block=3, noise=0.4, num_samples=500)
# Convert the iterator to a list of tuples
data = list(it)
# Separate the arrays and the integers
arrays, ints = zip(*data)
# Convert arrays to a 2D array (assuming all arrays have the same length)
array_data = np.vstack(arrays)
# Create the DataFrame
df = pd.DataFrame(array_data)
df['class'] = ints
df.to_csv('./data/generated_data.csv', index=False)
WIN_SIZE = 50
all_classes = np.array(df)[:, -1]
col1, col2 = st.columns(2)
st.markdown("")
btn1, btn2 = st.columns(2)
# Modify parameters
with btn1:
    with st.popover(":gear: Modify parameters"):
        st.write("""
        :gear: Modify the test parameters
        """)
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
            value=WIN_SIZE,
            placeholder="Window size",
        )

        metric_input = st.selectbox(
            "Choose the detection metric",
            [
                "Wasserstein order 1",
                "Wasserstein order 2",
                "Regularized Wasserstein",
            ],
            index=1,
        )

        cost_input = st.selectbox(
            "Choose the cost function",
            [
                "Euclidean",
                "Standardized Euclidean",
                "Mahalanobis",
            ],
            index=1,
        )

        if metric_input == "Wasserstein order 1":
            ot_metric = relio.OTMetric.WASSERSTEIN1
        elif metric_input == "Wasserstein order 2":
            ot_metric = relio.OTMetric.WASSERSTEIN2
        elif metric_input == "Regularized Wasserstein":
            ot_metric = relio.OTMetric.SINKHORN

        if cost_input == "Euclidean":
            cost_function = relio.CostFunction.EUCLIDEAN
        elif cost_input == "Standardized Euclidean":
            cost_function = relio.CostFunction.SEUCLIDEAN
        elif cost_input == "Mahalanobis":
            cost_function = relio.CostFunction.MAHALANOBIS

        alert_thold = st.number_input(
            "Enter the alert percentage",
            min_value=1,
            value=5,
            placeholder="Alert percentage",
        )

        detect_thold = st.number_input(
            "Enter the detection percentage",
            min_value=1,
            value=10,
            placeholder="Detection percentage",
        )

        stblty_thold = st.number_input(
            "Enter the stability threshold",
            min_value=1,
            value=3,
            placeholder="Stability threshold",
        )

# API initialization
api = relio.RelioApi(window_size, alert_thold, detect_thold,
                     ot_metric, cost_function, stblty_thold, df, 0)

ref_dist = []
for i in range(window_size):
    ref_dist.append(df.iloc[i])
first_concept = relio.Concept(1, np.array(ref_dist))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window = []

drift_impacts = []
adapt_perform = []

ref_dist_X = np.array(ref_dist)[:, :-1]
ref_dist_y = np.array(ref_dist)[:, -1].astype(int)
all_classes = np.unique(np.array(df)[:, -1].astype(int))

with col1:
    st.markdown(
        f"""
        :small_red_triangle_down: Window size : ***{window_size} data points***
        """,
        help="This is the :blue-background[number of data points] considered "
             "for calculating the drift metric.",
    )
    st.markdown(
        f"""
        :small_red_triangle_down: Detection metric : ***{metric_input}***
        """,
        help="This is the metric based on optimal transport to "
             ":blue-background[compare data distributions] in order to "
             "detect drift. Optimal transport provides a variety of metrics. "
             "We selected those most commonly used in the literature.",
    )
    st.markdown(
        f"""
        :small_red_triangle_down: Cost function : ***{cost_input}***
        """,
        help="This is a :blue-background[distance computed between data pairs]"
             "from two distributions, used by optimal transport metrics.",
    )
    st.markdown(
        f"""
        :small_red_triangle_down: Model type : ***{model_type}***
        """,
        help="Specifies whether the model used is "
             ":blue-background[supervised or unsupervised].",
    )

with col2:
    st.markdown(
        f"""
        :small_red_triangle_down: Alert percentage : ***{alert_thold}%***
        """,
        help="This is the :blue-background[distribution shift percentage] "
             "above which an alert is triggered. For example, if the "
             "comparison metric increases by 20%, an alert is raised.",
    )
    st.markdown(
        f"""
        :small_red_triangle_down: Detection percentage : ***{detect_thold}%***
        """,
        help="This is the :blue-background[distribution shift percentage] "
             "above which drift is confirmed. For example, if the "
             "comparison metric increases by 50%, drift is detected.",
    )
    st.markdown(
        f"""
        :small_red_triangle_down: Stability threshold : ***{stblty_thold}
         windows***
        """,
        help="This is the :blue-background[number of windows] used to declare "
             "that the data are :blue-background[stable within one "
             "distribution], meaning no drift is present.",
    )

pc1 = pca.fit_transform(df.iloc[:, :-1])
with btn2:
    button = st.button(":arrow_forward: Start ", type="primary")
if model_type == "Supervised - Stochastic Gradient Descent":
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'max_iter': [1000, 2000, 3000]
    }
    grid_search = GridSearchCV(estimator=SGDClassifier(
    ), param_grid=param_grid, cv=5, scoring='accuracy', error_score='raise')
    grid_search.fit(ref_dist_X, ref_dist_y)
    best_params = grid_search.best_params_
    model = SGDClassifier(**best_params, random_state=42)
    model.partial_fit(ref_dist_X, ref_dist_y, all_classes)
    drifted_model = SGDClassifier(**best_params, random_state=42)
    drifted_model.partial_fit(ref_dist_X, ref_dist_y, all_classes)
    METRIC_NAME = "Precision"
elif model_type == "Unsupervised - KMeans":
    silhouette_avg = []
    K = range(2, 11)
    for k in K:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(ref_dist_X)
        silhouette_avg.append(silhouette_score(ref_dist_X, cluster_labels))
    n = np.argmax(silhouette_avg)+2
    model = MiniBatchKMeans(n_clusters=n, random_state=42)
    drifted_model = MiniBatchKMeans(n_clusters=n, random_state=42)
    model = model.fit(ref_dist_X)
    cluster_labels_model = model.labels_
    drifted_model = drifted_model.fit(ref_dist_X)
    cluster_labels_drift = drifted_model.labels_
    adapt_perform.append(silhouette_score(ref_dist_X, cluster_labels_model))
    drift_impacts.append(silhouette_score(ref_dist_X, cluster_labels_drift))
    METRIC_NAME = "Silhouette Score"
if button:
    st.toast("API Initialisation ...", icon="⏳")

    st.write("""
    ##### :bar_chart: Data distribution evolution :
    """)
    chart = st.empty()
    st.write(f"""
       🔻 Quality of presentation of axis 1 =  **{(
           pca.explained_variance_ratio_[0] +
           pca.explained_variance_ratio_[1]) * 100:.2f}%**
    """)
    st.write(f"""
    ##### 	:chart_with_upwards_trend: {metric_input} evolution between the
    # reference window and the current window :
    """)
    distances = st.empty()

    st.write(f"""
    ##### 	📉 Drift Impact - {METRIC_NAME} Evolution:
    """)
    metric_chart = st.empty()

    st.divider()

    st.write("""
            ### :clock1: Detected drifts history:
    """)
    for i in range(window_size, len(df)+1):
        # Plot the data from the start to the current point
        chart.line_chart(pc1[:i])
        current_window.append(df.iloc[i-1])
        if len(current_window) == window_size:
            api.set_curr_win(np.array(current_window))
            api.monitor_drift()

            win_X = np.array(current_window)[:, :-1]
            win_y = np.array(current_window)[:, -1].astype(int)

            if model_type == "Supervised - Stochastic Gradient Descent":
                y_pred = model.predict(win_X)
                metric = accuracy_score(y_pred, win_y)

                y_pred_drift = drifted_model.predict(win_X)
                drifted_metric = accuracy_score(y_pred_drift, win_y)

            if api.get_action() == 0:
                drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(
                    f""":red[A drift is detected in data with index {
                        i + 1 - window_size} at {drift_time}]""", icon="⚠️")
                st.error(
                    f"""A drift is detected in data with index {
                        i + 1 - window_size} at {drift_time}""", icon="⚠️")
                drift_type = api.identify_type()
                if drift_type is not None:
                    if drift_type == relio.DriftType.GRADUAL:
                        st.toast(
                            ':blue[The drift type is: Gradual]', icon="📌")
                        st.info('The drift type is: Gradual', icon="📌")
                    elif drift_type == relio.DriftType.SUDDEN:
                        st.toast(
                            ':blue[The drift type is: Sudden]', icon="📌")
                        st.info('The drift type is: Sudden', icon="📌")
                    elif drift_type == relio.DriftType.RECURRENT:
                        st.toast(
                            ':blue[The drift type is: Recurrent]', icon="📌")
                        st.info('The drift type is: Recurrent', icon="📌")
                    elif drift_type == relio.DriftType.INCREMENTAL:
                        st.toast(
                            ':blue[The drift type is: Incremental]',
                            icon="📌")
                        st.info('The drift type is: Incremental', icon="📌")
                api.reset_retrain_model()

                train_X = np.concatenate((ref_dist_X, win_X))
                train_y = np.concatenate((ref_dist_y, win_y))
                if model_type == "Supervised - Stochastic Gradient Descent":
                    model.fit(win_X, win_y)
                elif model_type == "Unsupervised - KMeans":
                    silhouette_avg = []
                    K = range(2, 11)
                    for k in K:
                        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
                        cluster_labels = kmeans.fit_predict(train_X)
                        silhouette_avg.append(
                            silhouette_score(train_X, cluster_labels))
                    n = np.argmax(silhouette_avg)+2
                    model = MiniBatchKMeans(n_clusters=n, random_state=42)
                    model = model.fit(train_X)

                ref_dist_X = win_X
                ref_dist_y = win_y
            elif api.get_action() == 1:
                alert_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(
                    f"""Alert: A minor change in distribution has occurred in
                    data with index {i+1-window_size} at {alert_time}!""",
                    icon="❗")
                st.warning(
                    f"""Alert: A minor change in distribution has occurred in
                    data with index {i+1-window_size} at {alert_time}!""",
                    icon="❗")
                train_X = np.concatenate((ref_dist_X, win_X))
                train_y = np.concatenate((ref_dist_y, win_y))
                if model_type == "Supervised - Stochastic Gradient Descent":
                    model.partial_fit(win_X, win_y, classes=np.unique(win_y))
                elif model_type == "Unsupervised - KMeans":
                    model.partial_fit(train_X)

                api.reset_partial_fit()

            distances_data = pd.DataFrame(
                api.get_distances()[:i], columns=['distance'])
            distances_data['alert'] = api.get_alert_thold()
            distances_data['detection'] = api.get_detect_thold()
            distances.line_chart(distances_data, color=[
                                 "#338AFF", "#FFAC1C", "#FF0D0D"])

            if model_type == "Unsupervised - KMeans":
                labels = model.predict(win_X)
                labels_drift = drifted_model.predict(win_X)
                metric = silhouette_score(win_X, labels)
                drifted_metric = silhouette_score(win_X, labels_drift)
            adapt_perform.append(metric)
            drift_impacts.append(drifted_metric)

            metric_data = pd.DataFrame()
            metric_data['with adaptation'] = adapt_perform[:i]
            metric_data['without adaptation'] = drift_impacts[:i]
            metric_chart.line_chart(
                metric_data, color=["#338AFF", "#FF0D0D"])

            current_window = []
        drift_type = api.identify_type()
        if drift_type is not None:
            if drift_type == relio.DriftType.GRADUAL:
                st.toast(':blue[The drift type is: Gradual]', icon="📌")
                st.info('The drift type is: Gradual', icon="📌")
            elif drift_type == relio.DriftType.SUDDEN:
                st.toast(':blue[The drift type is: Sudden]', icon="📌")
                st.info('The drift type is: Sudden', icon="📌")
            elif drift_type == relio.DriftType.RECURRENT:
                st.toast(':blue[The drift type is: Recurrent]', icon="📌")
                st.info('The drift type is: Recurrent', icon="📌")
            elif drift_type == relio.DriftType.INCREMENTAL:
                st.toast(
                    ':blue[The drift type is: Incremental]', icon="📌")
                st.info('The drift type is: Incremental', icon="📌")
