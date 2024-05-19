import streamlit as st
import pandas as pd
import OT2D_API as ot2d
import time
import numpy as np



st.write("""
## :warning: Optimal Transport Drift Detection API
Optimal Transport Drift Detection (OT2D) est une interface de programmation d'application qui permet de détecter le concept drift causé par le changement de distribution dans les flux de données en temps réel.
""")
cd_container = st.container(border=True)
cd_container.write("""
### :question: Qu'est-ce que le concept drift?
Le concept drift se produit lorsque la distribution des données change avec le temps, ce qui rend le modèle d'apprentissage automatique supervisé ou non supervisé obsolète.
         """)

ot_container = st.container(border=True)
ot_container.write("""
### :question: Qu'est-ce que le transport optimal?
Le Transport Optimal est une méthode mathématique qui permet de trouver la meilleure correspondance entre deux distributions de probabilité. Il offre des métriques fiable pour comparer les distributions de données.
                  """)
st.divider()
st.write("""
### ▶️ Simulations
""")
st.page_link("pages/1 Drift Soudain.py", label="  Drift Soudain", icon="🔻")
st.page_link("pages/2 Drift Graduel.py", label="  Drift Graduel", icon="🔻")
st.page_link("pages/3 Drift Incremental.py", label="  Drift Incremental", icon="🔻")
st.page_link("pages/4 Drift Recurrent.py", label="  Drift Recurrent", icon="🔻")
st.divider()
st.write("""
### ▶️ Tests sur des datasets
""")
st.page_link("pages/5 Datasets Synthétiques.py", label="  Datasets Synthétiques", icon="🔻")
st.page_link("pages/6 Datasets Réels.py", label="  Datasets Réels", icon="🔻")

st.divider()
st.write("""
### ▶️ Application sur un cas d'étude
""")
st.page_link("pages/7 Application : Réseaux sociaux.py", label="  Réseaux Sociaux", icon="🔻")
