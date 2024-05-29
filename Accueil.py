import streamlit as st
import pandas as pd
import RELIO_API as relio
import time
import numpy as np
st.set_page_config(layout="wide")

st.write("""
# RELIO : Reliable Drift Detection API
         """)
cd_container = st.container(border=True)
cd_container.write("""
### :question: Qu'est-ce que le concept drift?
Le concept drift se produit lorsque :red-background[la distribution des données change avec le temps], ce qui rend le modèle d'apprentissage automatique supervisé ou non supervisé :red-background[obsolète].
         """)

cd_container = st.container(border=True)
cd_container.write("""
### :bulb: RELIO API
RELIO API est une interface de programmation d'application qui permet de :blue-background[détecter le concept drift] causé par le changement de distribution dans les flux de données en temps réel, en utilisant le :blue-background[transport optimal] pour quantifier le cout minimale pour transporter une distribution à une autre.
""")
st.divider()
st.write("""
### ▶️ Documentation
""")
st.page_link("pages/1 Documentation.py", label="Documentation", icon="📑")
st.divider()
st.write("""
### ▶️ Simulation des différents types de drifts
""")
st.page_link("pages/2 Simulation : Drift Soudain.py", label=" Simulation : Drift Soudain", icon="🔻")
st.page_link("pages/3 Simulation : Drift Graduel.py", label=" Simulation : Drift Graduel", icon="🔻")
st.page_link("pages/4 Simulation : Drift Recurrent.py", label=" Simulation : Drift Recurrent", icon="🔻")
st.page_link("pages/5 Simulation : Drift Incremental.py", label=" Simulation : Drift Incremental", icon="🔻")
st.divider()
st.write("""
### ▶️ Tests sur des datasets
""")
st.page_link("pages/6 Tests : Datasets Synthétiques.py", label=" Tests : Datasets Synthétiques", icon="🔻")
st.page_link("pages/7 Tests : Datasets Réels.py", label=" Tests : Datasets Réels", icon="🔻")

st.divider()
st.write("""
### ▶️ Comparaison avec d'autres solutions
""")
st.page_link("pages/8 Comparaison.py", label=" Comparaison", icon="🔻")
