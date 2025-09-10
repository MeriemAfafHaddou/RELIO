import datetime

import numpy as np
import pandas as pd
import streamlit as st

import relio_api as relio

st.set_page_config(
    page_title="Application to social media",
    page_icon="images/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.write("""
# Application : Drift detection in social media networks
""")
st.write("""### Test :""")
# Dataset choice
option = st.selectbox(
    ":bar_chart: Which dataset would you like to pick ?",
    ("Cannes 2013", "Digg", "Cell"),
)

if option == "Cannes 2013":
    df1 = pd.read_csv(
        "data/graphes/Cannes/Cannes2013_snapshot_0.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df2 = pd.read_csv(
        "data/graphes/Cannes/Cannes2013_snapshot_1.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df3 = pd.read_csv(
        "data/graphes/Cannes/Cannes2013_snapshot_2.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df4 = pd.read_csv(
        "data/graphes/Cannes/Cannes2013_snapshot_3.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df5 = pd.read_csv(
        "data/graphes/Cannes/Cannes2013_snapshot_4.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df6 = pd.read_csv(
        "data/graphes/Cannes/Cannes2013_snapshot_5.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df7 = pd.read_csv(
        "data/graphes/Cannes/Cannes2013_snapshot_6.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df8 = pd.read_csv(
        "data/graphes/Cannes/Cannes2013_snapshot_7.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df = [df1, df2, df3, df4, df5, df6, df7, df8]
    ALERT_INIT = 0.43
    DETECT_INIT = 0.46
elif option == "Cell":
    df1 = pd.read_csv(
        "data/graphes/cell/real.t01.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df2 = pd.read_csv(
        "data/graphes/cell/real.t02.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df3 = pd.read_csv(
        "data/graphes/cell/real.t03.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df4 = pd.read_csv(
        "data/graphes/cell/real.t04.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df5 = pd.read_csv(
        "data/graphes/cell/real.t05.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df6 = pd.read_csv(
        "data/graphes/cell/real.t06.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df7 = pd.read_csv(
        "data/graphes/cell/real.t07.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df8 = pd.read_csv(
        "data/graphes/cell/real.t08.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df9 = pd.read_csv(
        "data/graphes/cell/real.t09.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df10 = pd.read_csv(
        "data/graphes/cell/real.t010.edges",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
    ALERT_INIT = 0.15
    DETECT_INIT = 0.17
elif option == "Digg":
    df1 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_1.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df2 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_2.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df3 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_3.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df4 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_4.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df5 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_5.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df6 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_6.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df7 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_7.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df8 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_8.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df9 = pd.read_csv(
        "data/graphes/digg/Digg_snapshot_9.edgelist.txt",
        delimiter=r"\s+",
        header=None,
        usecols=[0, 1],
    )
    df = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
    ALERT_INIT = 0.5
    DETECT_INIT = 0.7
col1, col2 = st.columns(2)
st.markdown("")
btn1, btn2 = st.columns(2)
# Modify parameters
with btn1:
    with st.popover(":gear: Modify parameters"):
        st.write("""
        :gear: Modify test parameters
        """)
        window_size = st.number_input(
            "Enter the window size",
            min_value=1,
            value=24,
            placeholder="Window size",
        )
        metric_input = st.selectbox(
            "Select the detection metric",
            [
                "Wasserstein of order 1",
                "Wasserstein of order 2",
                "Regularized Wasserstein",
            ],
            index=1,
        )
        cost_input = st.selectbox(
            "Select the cost function",
            ["Euclidean", "Standardized Euclidean", "Mahalanobis"],
            index=1,
        )
        if metric_input == "Wasserstein of order 1":
            ot_metric = relio.OTMetric.WASSERSTEIN1
        elif metric_input == "Wasserstein of order 2":
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
            "Enter the alert threshold (%)",
            value=ALERT_INIT,
            placeholder="Alert threshold (%)",
        )
        detect_thold = st.number_input(
            "Enter the detection threshold (%)",
            value=DETECT_INIT,
            placeholder="Detection threshold (%)",
        )
        stblty_thold = st.number_input(
            "Enter the stability threshold",
            min_value=1,
            value=3,
            placeholder="Stability threshold",
        )


api = relio.RelioApi(24, alert_thold, detect_thold, ot_metric, cost_function,
                     stblty_thold, df, 1,)
ref_dist = df[0]
first_concept = relio.Concept(1, np.array(ref_dist))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window = []
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
with btn2:
    button = st.button(":arrow_forward: Start ", type="primary")
if button:
    st.toast("Initializing the API...", icon="⏳")

    st.write(
        f"""
        ##### :chart_with_upwards_trend: Evolution of the {metric_input}
        distance between the reference distribution and the current window:
        """
    )
    distances = st.empty()

    st.divider()

    st.write("""
            ### :clock1: History of detected drifts:
    """)
    i = 0
    for current_window in df[1:]:
        i += 1
        api.set_curr_win(np.array(current_window))
        api.monitor_drift()

        if api.get_action() == 0:
            drift_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.toast(
                ":red[A drift is detected in data with index "
                f"{i + 1 - window_size} at {drift_time}]",
                icon="⚠️",
            )
            st.error(
                "A drift is detected in data with index "
                f"{i + 1 - window_size} at {drift_time}",
                icon="⚠️",
            )
            drift_type = api.identify_type()
            if drift_type is not None:
                if drift_type == relio.DriftType.GRADUAL:
                    st.toast(":blue[The drift type is: Gradual]",
                             icon="📌")
                    st.info("The drift type is: Gradual", icon="📌")
                elif drift_type == relio.DriftType.SUDDEN:
                    st.toast(":blue[The drift type is: Sudden]", icon="📌")
                    st.info("The drift type is: Sudden", icon="📌")
                elif drift_type == relio.DriftType.RECURRENT:
                    st.toast(
                        ":blue[The drift type is: Recurrent]", icon="📌")
                    st.info("The drift type is: Recurrent", icon="📌")
                elif drift_type == relio.DriftType.INCREMENTAL:
                    st.toast(
                        ":blue[The drift type is: Incremental]", icon="📌")
                    st.info("The drift type is: Incremental", icon="📌")
            api.reset_retrain_model()

            ref_dist = current_window
        elif api.get_action() == 1:
            alert_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.toast(
                f"Alert: A minor change in distribution has occurred in "
                f"data with index {i + 1 - window_size} at {alert_time}!",
                icon="❗",
            )
            st.warning(
                f"Alert: A minor change in distribution has occurred in "
                f"data with index {i + 1 - window_size} at {alert_time}!",
                icon="❗",
            )
            api.reset_partial_fit()
        distances_data = pd.DataFrame(
            api.get_distances()[:i], columns=["distance"])
        distances_data["alert"] = api.get_alert_thold()
        distances_data["detection"] = api.get_detect_thold()
        distances.line_chart(
            distances_data, color=["#338AFF", "#FFAC1C", "#FF0D0D"]
        )
        drift_type = api.identify_type()
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
                st.toast(":blue[The drift type is: Incremental]", icon="📌")
                st.info("The drift type is: Incremental", icon="📌")
