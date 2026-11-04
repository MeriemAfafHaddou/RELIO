import streamlit as st

st.logo("images/logo.png")
st.set_page_config(
    page_title="Relio - Accueil",
    page_icon="images/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.write("""
# 🏠 RELIO - Home
         """)
st.markdown(""" """)


cd_container = st.container(border=True)
with cd_container:
    st.write(""" ### :bulb: RELIO API""")
    col1, col2 = st.columns([2, 1], gap="large")
    col1.write("""
      RELIO API is a programming interface that allows you to detect concept
      drift caused by distribution changes in real-time data streams, using
      optimal transport to measure the minimum cost of moving from one data
      distribution to another.
      """)
    col2.image("images/logo.png", width=300)
st.write("""
### ▶️ Documentation
""")
st.page_link("pages/1_documentation.py", label="Documentation", icon="📑")
st.divider()
st.write("""
### ▶️ Simulation of different types of drifts
""")
st.page_link("pages/2_simulation_iris_sudden.py",
             label=" Simulation : Iris Sudden", icon="🔻")
st.page_link("pages/3_simulation_iris_recurrent.py",
             label=" Simulation : Iris Recurrent", icon="🔻")
st.page_link("pages/4_simulation_iris_incremental.py",
             label=" Simulation : Iris Incremental", icon="🔻")
st.page_link("pages/5_simulation_iris_gradual.py",
             label=" Simulation : Iris Gradual", icon="🔻")

st.divider()

st.write("""
### ▶️ Tests on datasets
""")
st.page_link("pages/6_synthetic_data.py",
             label=" Tests : Synthetic Datasets", icon="🔻")
st.page_link("pages/7_real_data.py",
             label=" Tests : Real Datasets", icon="🔻")
st.page_link("pages/8_application_on_social_modia.py",
             label=" Application : Social Media", icon="🔻")

st.divider()
st.write("""
### ▶️ Comparison with other solutions
""")
st.page_link("pages/9_comparaison.py", label=" Comparaison", icon="🔻")
