import streamlit as st
import pandas as pd
import OT2D_API as ot2d
import time
import numpy as np
import datetime

st.write("""
# OT2D : Simulation d'un drift récurrent
""")
with st.expander(":blue[:question: Qu'est-ce qu'un drift récurrent ?]",expanded=False):
    st.write('''
        Le concept drift récurrent se réfère à des changements de données qui réapparaissent
après un certain temps, sans suivre nécessairement un schéma périodique comme indiqué sur
la figure : 
             
    ''')
    st.image('images/recurrent.png')

st.write("""
         ### Simulation : 
""")
df = pd.read_csv("data/iris_recurrent.csv")


with st.popover(":gear: Modifier les paramètres"):
    st.write("""
     :gear: Modifier les paramètres de la simulation 
     """)
    window_size = st.number_input('Introduire la taille de la fenêtre', min_value=1, value=20, placeholder="Taille de la fenêtre")
    metric_input=st.selectbox('Choisir la métrique de détection', ['Wasserstein d\'ordre 1', 'Wasserstein d\'ordre 2', 'Wasserstein régularisé'], index=1)
    cost_input=st.selectbox('Choisir la fonction de coût', ['Euclidienne', 'Euclidienne Standarisée', 'Mahalanobis'], index=1)
    if metric_input == 'Wasserstein d\'ordre 1':
        ot_metric = ot2d.OTMetric.WASSERSTEIN1
    elif metric_input == 'Wasserstein d\'ordre 2':
        ot_metric = ot2d.OTMetric.WASSERSTEIN2
    elif metric_input == 'Wasserstein régularisé':
        ot_metric = ot2d.OTMetric.SINKHORN

    if cost_input == 'Euclidienne':
        cost_function = ot2d.CostFunction.EUCLIDEAN
    elif cost_input == 'Euclidienne Standarisée':
        cost_function = ot2d.CostFunction.SEUCLIDEAN
    elif cost_input == 'Mahalanobis':    
        cost_function = ot2d.CostFunction.MAHALANOBIS
    alert_thold=st.number_input('Introduire le seuil d\'alerte', min_value=0.1, value=1.5, placeholder="Seuil d'alerte")
    detect_thold=st.number_input('Introduire le seuil de détection', min_value=0.1, value=1.7, placeholder="Seuil de détection")
    stblty_thold=st.number_input('Introduire le seuil de stabilité', min_value=1, value=6, placeholder="Seuil de stabilité")

api=ot2d.OT2D(window_size, alert_thold, detect_thold, ot_metric, cost_function, stblty_thold )
ref_dist=[]
for i in range(window_size):
    ref_dist.append(df.iloc[i])
first_concept=ot2d.Concept(1, np.array(ref_dist))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window=[]
col1, col2 = st.columns(2)
with col1:
    st.write(f"""
    :small_red_triangle_down: Taille de la fenêtre : ***{window_size} Données*** \n
    :small_red_triangle_down: Métrique de détection : ***{metric_input}*** \n
    :small_red_triangle_down: Fonction de coût : ***{cost_input}***
         """)
with col2:
    st.write(f"""
    :small_red_triangle_down: Seuil d'alerte : ***{alert_thold}*** \n
    :small_red_triangle_down: Seuil de détection : ***{detect_thold}*** \n
    :small_red_triangle_down: Seuil de stabilité : ***{stblty_thold} fenêtres***
         """)
button=st.button(":arrow_forward: Lancer la simulation", type="primary")
if button:
    st.toast("Initialisation de l'API en cours...", icon="⏳")
    st.write("""
    ##### :bar_chart: Évolution de la distribution de données : 
    """)
    chart = st.empty()
    st.write(f"""
    ##### 	:chart_with_upwards_trend: Évolution de la distance de {metric_input} entre la distribution de référence et la fenêtre courante  : 
    """)
    distances=st.empty()
    st.divider()
    st.write("""
            ### :clock1: Historique des drifts détectés: 
    """)
    for i in range(window_size, len(df)+1):
        # Plot the data from the start to the current point
        chart.line_chart(df[['petal_length', 'petal_width']].iloc[:i])
        current_window.append(df.iloc[i-1])
        if len(current_window) == window_size:
            api.set_curr_win(np.array(current_window))
            api.monitorDrift()
            distances_data=pd.DataFrame(api.get_distances()[:i], columns=['Distance'])
            distances_data['Alerte']=alert_thold
            distances_data['Détection']=detect_thold
            distances.line_chart(distances_data, color=["#FFAC1C","#338AFF", "#FF0D0D"])

            if(api.get_action()==0):
                drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(f":red[Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}]", icon="⚠️")
                st.error(f"Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}", icon="⚠️")
                drift_type=api.identifyType()
                if(drift_type != None):
                    if drift_type == ot2d.DriftType.GRADUAL:
                        st.toast(f':blue[Le type de drift est : Graduel]', icon="📌")
                        st.info(f'Le type de drift est : Graduel', icon="📌")
                    elif drift_type == ot2d.DriftType.SUDDEN:
                        st.toast(f':blue[Le type de drift est : Soudain]', icon="📌")
                        st.info(f'Le type de drift est : Soudain', icon="📌")
                    elif drift_type == ot2d.DriftType.RECURRENT:
                        st.toast(f':blue[Le type de drift est : Récurrent]', icon="📌")
                        st.info(f'Le type de drift est : Récurrent', icon="📌")
                    elif drift_type == ot2d.DriftType.INCREMENTAL:
                        st.toast(f':blue[Le type de drift est : Incrémental]', icon="📌")
                        st.info(f'Le type de drift est : Incrémental', icon="📌")
                api.reset_retrain_model()
            elif (api.get_action()==1):
                st.toast(f"Alerte : Un petit changement de distribution s'est produit !", icon="❗")
                st.warning(f"Alerte : Un petit changement de distribution s'est produit !", icon="❗")

            current_window=[]
        drift_type=api.identifyType()
        if(drift_type != None):
            if drift_type == ot2d.DriftType.GRADUAL:
                st.toast(f':blue[Le type de drift est : Graduel]', icon="📌")
                st.info(f'Le type de drift est : Graduel', icon="📌")
            elif drift_type == ot2d.DriftType.SUDDEN:
                st.toast(f':blue[Le type de drift est : Soudain]', icon="📌")
                st.info(f'Le type de drift est : Soudain', icon="📌")
            elif drift_type == ot2d.DriftType.RECURRENT:
                st.toast(f':blue[Le type de drift est : Récurrent]', icon="📌")
                st.info(f'Le type de drift est : Récurrent', icon="📌")
            elif drift_type == ot2d.DriftType.INCREMENTAL:
                st.toast(f':blue[Le type de drift est : Incrémental]', icon="📌")
                st.info(f'Le type de drift est : Incrémental', icon="📌")
        # Pause for a moment
        time.sleep(0.05)