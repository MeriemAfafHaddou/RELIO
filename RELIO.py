import streamlit as st
import pandas as pd
import RELIO_API as relio
import time
import numpy as np


st.logo("images/logo.png")
st.set_page_config(
   page_title="Relio - Accueil",
   page_icon="images/icon.png",
   layout="wide",
   initial_sidebar_state="expanded",
)
st.write("""
# 🏠 RELIO - Accueil
         """)
st.markdown(""" """)

cd_container = st.container(border=True)
cd_container.write("""
### :question: Qu'est-ce que le concept drift?
Le concept drift se produit lorsque :blue-background[la distribution des données change avec le temps], ce qui rend le modèle d'apprentissage automatique supervisé ou non supervisé :blue-background[obsolète].
                   D'où la nécessité de mettre en place une solution de détection de drift fiable et robuste.
         """)
st.markdown(""" """)


cd_container = st.container(border=True)
with cd_container:
      st.write(""" ### :bulb: RELIO API""")
      col1, col2=st.columns([2,1], gap="large")
      col1.write("""
      RELIO API est une interface de programmation d'application qui permet de :blue-background[détecter le concept drift] causé par le changement de distribution dans les flux de données en temps réel, en utilisant le :blue-background[transport optimal] pour quantifier le cout minimale pour transporter une distribution à une autre.
      """)
      col2.image("images/logo.png", width=300)
st.divider()
st.write("""
### ▶️ Documentation
""")
st.page_link("pages/1 Documentation.py", label="Documentation", icon="📑")
st.divider()
st.write("""
### ▶️ Génération de données
""")
st.page_link("pages/2 Génération de données.py", label=" Génération de données", icon="🔻")

st.divider()
st.write("""
### ▶️ Simulation des différents types de drifts
""")
st.page_link("pages/3 Simulation : Drift Soudain.py", label=" Simulation : Drift Soudain", icon="🔻")
st.page_link("pages/4 Simulation : Drift Graduel.py", label=" Simulation : Drift Graduel", icon="🔻")
st.page_link("pages/5 Simulation : Drift Recurrent.py", label=" Simulation : Drift Recurrent", icon="🔻")
st.page_link("pages/6 Simulation : Drift Incremental.py", label=" Simulation : Drift Incremental", icon="🔻")
st.divider()

st.write("""
### ▶️ Tests sur des datasets
""")
st.page_link("pages/7 Datasets Synthétiques.py", label=" Tests : Datasets Synthétiques", icon="🔻")
st.page_link("pages/8 Datasets Réels.py", label=" Tests : Datasets Réels", icon="🔻")

st.divider()
st.write("""
### ▶️ Comparaison avec d'autres solutions
""")
st.page_link("pages/9 Comparaison.py", label=" Comparaison", icon="🔻")
