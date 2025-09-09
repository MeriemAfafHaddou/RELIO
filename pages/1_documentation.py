import streamlit as st

st.logo("images/logo.png")
st.set_page_config(
    page_title="Relio - Documentation",
    page_icon="images/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.write(""" # 📑 RELIO - Documentation \n
         """)
st.markdown(""" """)
relio_container = st.container(border=True)
relio_container.write("""
### :question: What is ReliO?
ReliO (RELIable Outcomes) is an API defined by its
:blue-background[**reliable results**] for
:blue-background[**concept drift detection**] based on
:blue-background[**optimal transport**].
It allows detecting concept drift caused by distribution shifts in
real-time data streams.
""")
st.markdown(""" """)

ot_container = st.container(border=True)
ot_container.write("""
### :question: What is Optimal Transport?
Optimal Transport is a mathematical method used to find the best alignment
between two probability distributions.
It provides reliable metrics for comparing data distributions.
""")

st.markdown(""" """)


param_container = st.container(border=True)
param_container.write("""
#### :gear: Simulation Parameters """)
param_container.markdown("""
    :small_red_triangle_down: **Window size**: the :blue-background[number
    of data points] considered for calculating the drift metric. \n
    :small_red_triangle_down: **Detection metric**: the metric based on
    optimal transport to :blue-background[compare data distributions] in
    order to detect drift, by :blue-background[quantifying the minimal
    cost of moving one distribution to another]. \n
    :small_red_triangle_down: **Cost function**: a :blue-background[distance
    computed between data pairs] from two distributions, used by optimal
    transport metrics. Options include Euclidean distance, standardized
    Euclidean, and Mahalanobis distance. \n
    :small_red_triangle_down: **Alert percentage**: the
    :blue-background[distribution shift percentage] above which an alert
    is triggered. For example, if the comparison metric increases by 20%,
    an alert is raised. \n
    :small_red_triangle_down: **Detection percentage**: the
    :blue-background[distribution shift percentage] above which drift is
    confirmed. For example, if the comparison metric increases by 50%,
    drift is detected. \n
    :small_red_triangle_down: **Stability threshold**: the
    :blue-background[number of windows] used to declare that the data are
    :blue-background[stable within one distribution], meaning no drift is
    present. \n
""")

param_container.markdown(""" """)
