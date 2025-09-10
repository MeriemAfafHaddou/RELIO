import datetime

import numpy as np
import pandas as pd
import streamlit as st
from frouros.callbacks.batch import PermutationTestDistanceBased
from frouros.datasets.synthetic import SEA
from frouros.detectors.data_drift import EMD, JS
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import relio_api as relio

st.logo("images/logo.png")
st.set_page_config(
    page_title="Comparaison",
    page_icon="images/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.write("""
# Comparaison between RELIO and existing methods
""")
st.markdown("####")
pca = PCA(n_components=1)

option = st.selectbox(
    ":bar_chart: Which dataset would you like to pick ?",
    (
        "Generated data",
        "Simulation : Iris Gradual",
        "Simulation : Iris Incremental",
        "Simulation : Iris Sudden",
        "Simulation : Iris Recurrent",
        "Synthetic : Insects Sudden",
        "Synthetic : Insects Incremental",
        "Ozone",
        "Asfault",
    ),
)

col1, col2 = st.columns(2)

if option == "Asfault":
    df = pd.read_csv("data/Asfault.csv", header=None)[:8000]
    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df[64])
    class_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )
    df = df.drop(64, axis="columns")
    df.columns = df.columns.astype(str)
    ALERT_INIT = 10
    DETECT_INIT = 20
    WIN_SIZE = 500
elif option == "Ozone":
    df = pd.read_csv("data/Ozone.csv", header=None)
    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df[72])
    class_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )
    df = df.drop(72, axis="columns")
    df.columns = df.columns.astype(str)
    ALERT_INIT = 50
    DETECT_INIT = 70
    WIN_SIZE = 150
elif option == "Simulation : Iris Sudden":
    df = pd.read_csv("data/iris_sudden.csv")
    ALERT_INIT = 20
    DETECT_INIT = 40
    WIN_SIZE = 50
elif option == "Simulation : Iris Gradual":
    df = pd.read_csv("data/iris_graduel.csv")
    ALERT_INIT = 20
    DETECT_INIT = 40
    WIN_SIZE = 20
elif option == "Simulation : Iris Recurrent":
    df = pd.read_csv("data/iris_recurrent.csv")
    ALERT_INIT = 5
    DETECT_INIT = 25
    WIN_SIZE = 20
elif option == "Simulation : Iris Incremental":
    df = pd.read_csv("data/iris_incremental.csv")
    ALERT_INIT = 5
    DETECT_INIT = 25
    WIN_SIZE = 40
elif option == "Synthetic : Insects Sudden":
    df = pd.read_csv("data/insects_sudden.csv", header=None)[9800:13800]
    ALERT_INIT = 50
    DETECT_INIT = 70
    WIN_SIZE = 200
elif option == "Synthetic : Insects Incremental":
    df = pd.read_csv("data/insects_incremental.csv", header=None)[32000:37000]
    ALERT_INIT = 20
    DETECT_INIT = 40
    WIN_SIZE = 200
elif option == "Generated data":
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
    df["class"] = ints
    WIN_SIZE = 50
    ALERT_INIT = 5
    DETECT_INIT = 10

# Modify parameters
st.markdown("")
btn1, btn2 = st.columns(2)
with btn1:
    with st.popover(":gear: Adjust parameters"):
        st.write("""
        :gear: Adjust test parameters
        """)
        window_size = st.number_input(
            "Enter the window size",
            min_value=1,
            value=WIN_SIZE,
            placeholder="Window size",
        )
        metric_input = st.selectbox(
            "Select the detection metric",
            [
                "Wasserstein order 1",
                "Wasserstein order 2",
                "Regularized Wasserstein",
            ],
            index=1,
        )
        cost_input = st.selectbox(
            "Select the cost function",
            ["Euclidean", "Standardized Euclidean", "Mahalanobis"],
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
            value=ALERT_INIT,
            placeholder="Alert percentage",
        )
        detect_thold = st.number_input(
            "Enter the detection percentage",
            min_value=1,
            value=DETECT_INIT,
            placeholder="Detection percentage",
        )
        stblty_thold = st.number_input(
            "Enter the stability threshold",
            min_value=1,
            value=4,
            placeholder="Stability threshold",
        )

pc1 = pca.fit_transform(df.iloc[:, :-1])

# API initialization
api = relio.RelioApi(
    window_size,
    alert_thold,
    detect_thold,
    ot_metric,
    cost_function,
    stblty_thold,
    df,
    0,
)

with col1:
    st.markdown(
        f"""
        :small_red_triangle_down: Window size: ***{window_size} data points***
        """,
        help="This is the :blue-background[number of data points] considered "
        "for calculating the drift metric.",
    )

    st.markdown(
        f"""
        :small_red_triangle_down: Detection metric: ***{metric_input}***
        """,
        help="This is the metric based on optimal transport to "
        ":blue-background[compare data distributions] in order to detect "
        "drift. Optimal transport offers a variety of metrics. We chose "
        "the ones most widely used in the literature.",
    )

    st.markdown(
        f"""
        :small_red_triangle_down: Cost function: ***{cost_input}***
        """,
        help="This is a :blue-background[distance computed between data pairs]"
        " from two distributions, used by optimal transport metrics.",
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
        "above which drift is detected. For example, if the comparison "
        "metric increases by 50%, drift is confirmed.",
    )

    st.markdown(
        f"""
        :small_red_triangle_down: Stability threshold : ***{stblty_thold}
        windows***
        """,
        help="This is the :blue-background[number of windows] used to declare "
        "that the data are :blue-background[stable within a "
        "distribution], meaning no drift is present.",
    )

fr_win = np.array([])
emd_detector = EMD(
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
js_detector = JS(
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
ALPHA = 0.05
fr_ref = pc1[:WIN_SIZE]
_ = emd_detector.fit(X=fr_ref)
_ = js_detector.fit(X=fr_ref)

ref_dist = []
for i in range(window_size):
    ref_dist.append(df.iloc[i])
first_concept = relio.Concept(1, np.array(ref_dist))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window = []
drifts = []
ref_dist_X = fr_ref
ref_dist_y = np.array(ref_dist)[:, -1].astype(int)
all_classes = np.unique(np.array(df)[:, -1].astype(int))
dist_emd = []
dist_js = []
with btn2:
    button = st.button(":arrow_forward: Lancer le test ", type="primary")
if button:
    st.toast("Initializing the API...", icon="⏳")
    st.write("""
    ##### :bar_chart: Evolution of the data distribution:
    """)
    chart = st.empty()
    st.write(
        f"""
        🔻 Quality of representation on axis 1 =
        **{pca.explained_variance_ratio_[0] * 100:.2f}%**
        """
    )

    st.divider()

    st.write("""
            ### :scales: Detection results comparaison
    """)
    relio_col, fr_col1, fr_col2 = st.columns(3)
    with relio_col:
        st.markdown("""##### RELIO """)
        st.write("""
        ##### 	:chart_with_upwards_trend: Relio metric evolution
        """)
        distances_relio = st.empty()
        st.divider()

    with fr_col1:
        st.markdown(
            """##### Frouros - EMD """,
            help="Earth Mover's Distance : 1-Wasserstein",
        )
        st.write("""
        ##### 	:chart_with_upwards_trend: EMD distance evolution :
        """)
        distances_emd = st.empty()
        st.divider()

    with fr_col2:
        st.markdown(
            """##### Frouros - JS """,
            help="Jensen-Shannon Distance: a distance based on Kullback-"
                "Leibler divergence",
        )
        st.write("""
            ##### :chart_with_upwards_trend: Evolution of the JS distance:
        """)

        distances_js = st.empty()
        st.divider()
    for i in range(window_size, len(df) + 1):
        # Plot the data from the start to the current point
        chart.line_chart(pc1[:i])
        current_window.append(df.iloc[i - 1])
        if len(current_window) == window_size:
            fr_win = pc1[i - window_size: i]
            api.set_curr_win(np.array(current_window))
            api.monitor_drift()

            distances_data = pd.DataFrame(
                api.get_distances()[:i], columns=["distance"])
            distances_data["alert"] = api.get_alert_thold()
            distances_data["detection"] = api.get_detect_thold()

            win_X = np.array(current_window)[:, :-1]
            win_y = np.array(current_window)[:, -1].astype(int)
            if api.get_action() == 0:
                drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(
                    ":red[A drift is detected in data with index "
                    f"{i + 1 - window_size} at {drift_time}]",
                    icon="⚠️",
                )
                with relio_col:
                    st.error(
                        " RELIO : A drift is detected in data with index "
                        f"{i + 1 - window_size} à {drift_time}",
                        icon="⚠️",
                    )
                drift_type = api.identify_type()
                with relio_col:
                    if drift_type is not None:
                        if drift_type == relio.DriftType.GRADUAL:
                            st.toast(":blue[The drift type is: Gradual]",
                                    icon="📌")
                            st.info("The drift type is: Gradual", icon="📌")
                        elif drift_type == relio.DriftType.SUDDEN:
                            st.toast(":blue[The drift type is: Sudden]",
                                     icon="📌")
                            st.info("The drift type is: Sudden", icon="📌")
                        elif drift_type == relio.DriftType.RECURRENT:
                            st.toast(
                                ":blue[The drift type is: Recurrent]",
                                icon="📌")
                            st.info("The drift type is: Recurrent", icon="📌")
                        elif drift_type == relio.DriftType.INCREMENTAL:
                            st.toast(
                                ":blue[The drift type is: Incremental]",
                                icon="📌")
                            st.info("The drift type is: Incremental",
                                    icon="📌")
                api.reset_retrain_model()

                ref_dist_X = win_X
                ref_dist_y = win_y
            elif api.get_action() == 1:
                alert_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(
                    f"Alert: A minor change in distribution has occurred in "
                    f"data with index {i + 1 - window_size} at {alert_time}!",
                    icon="❗",
                )
                with relio_col:
                    st.warning(
                        "Alert: A minor change in distribution has occurred "
                        f"in data with index {i + 1 - window_size} at "
                        f"{alert_time}!",
                        icon="❗",
                    )
                api.reset_partial_fit()

            with relio_col:
                distances_relio.line_chart(
                    distances_data, color=["#338AFF", "#FFAC1C", "#FF0D0D"]
                )

            emd_dist, callbacks_log = emd_detector.compare(X=fr_win)
            p_value = callbacks_log["permutation_test"]["p_value"]
            if p_value <= ALPHA:
                with fr_col1:
                    drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                    st.error(
                        " Frouros : A drift is detected in data with index "
                        f"{i + 1 - window_size} at {drift_time}",
                        icon="⚠️",
                    )
                    _ = emd_detector.fit(X=fr_win)
            with fr_col1:
                dist_emd.append(emd_dist)
                distances_emd.line_chart(dist_emd, color=["#338AFF"])
            js_dist, callbacks_log = js_detector.compare(X=fr_win)
            p_value = callbacks_log["permutation_test"]["p_value"]
            if p_value <= ALPHA:
                with fr_col2:
                    drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                    st.error(
                        " Frouros : A drift is detected in data with index "
                        f"{i + 1 - window_size} at {drift_time}",
                        icon="⚠️",
                    )
                    _ = js_detector.fit(X=fr_win)

            with fr_col2:
                dist_js.append(js_dist)
                distances_js.line_chart(dist_js, color=["#338AFF"])
            current_window = []
            fr_win = np.array([])

        drift_type = api.identify_type()
        with relio_col:
            if drift_type is not None:
                if drift_type == relio.DriftType.GRADUAL:
                    st.toast(":blue[The drift type is: Gradual]", icon="📌")
                    st.info("The drift type is: Gradual", icon="📌")
                elif drift_type == relio.DriftType.SUDDEN:
                    st.toast(":blue[The drift type is: Sudden]", icon="📌")
                    st.info("The drift type is: Sudden", icon="📌")
                elif drift_type == relio.DriftType.RECURRENT:
                    st.toast(":blue[The drift type is: Recurrent]", icon="📌")
                    st.info("The drift type is: Recurrent", icon="📌")
                elif drift_type == relio.DriftType.INCREMENTAL:
                    st.toast(":blue[The drift type is: Incremental]",
                             icon="📌")
                    st.info("The drift type is: Incremental", icon="📌")
