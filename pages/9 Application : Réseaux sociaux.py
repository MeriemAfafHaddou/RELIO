import streamlit as st
import pandas as pd
import numpy as np
import RELIO_API as relio
import datetime

st.set_page_config(
   page_title="Application aux réseaux sociaux",
   page_icon="images/icon.png",
   layout="wide",
   initial_sidebar_state="expanded",
)
st.write("""
# Application : Détection de drift dans les réseaux sociaux
""")
st.write("""
         ### Test : 
""")
# Dataset choice
option = st.selectbox(
    ":bar_chart: Quel dataset voulez vous choisir?",
    ("Cannes 2013","Digg", "Cell"))

if option=="Cannes 2013":
    df1=pd.read_csv('data/graphes/Cannes/Cannes2013_snapshot_0.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df2=pd.read_csv('data/graphes/Cannes/Cannes2013_snapshot_1.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df3=pd.read_csv('data/graphes/Cannes/Cannes2013_snapshot_2.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df4=pd.read_csv('data/graphes/Cannes/Cannes2013_snapshot_3.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df5=pd.read_csv('data/graphes/Cannes/Cannes2013_snapshot_4.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df6=pd.read_csv('data/graphes/Cannes/Cannes2013_snapshot_5.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df7=pd.read_csv('data/graphes/Cannes/Cannes2013_snapshot_6.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df8=pd.read_csv('data/graphes/Cannes/Cannes2013_snapshot_7.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df=[df1, df2, df3, df4, df5, df6, df7, df8]
    alert_init=0.43
    detect_init=0.46
elif option=="Cell":
    df1=pd.read_csv('data/graphes/cell/real.t01.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df2=pd.read_csv('data/graphes/cell/real.t02.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df3=pd.read_csv('data/graphes/cell/real.t03.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df4=pd.read_csv('data/graphes/cell/real.t04.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df5=pd.read_csv('data/graphes/cell/real.t05.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df6=pd.read_csv('data/graphes/cell/real.t06.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df7=pd.read_csv('data/graphes/cell/real.t07.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df8=pd.read_csv('data/graphes/cell/real.t08.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df9=pd.read_csv('data/graphes/cell/real.t09.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df10=pd.read_csv('data/graphes/cell/real.t010.edges', delimiter='\s+', header=None, usecols=[0, 1])
    df=[df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
    alert_init=0.15
    detect_init=0.17
elif option=="Digg":
    df1=pd.read_csv('data/graphes/digg/Digg_snapshot_1.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df2=pd.read_csv('data/graphes/digg/Digg_snapshot_2.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df3=pd.read_csv('data/graphes/digg/Digg_snapshot_3.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df4=pd.read_csv('data/graphes/digg/Digg_snapshot_4.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df5=pd.read_csv('data/graphes/digg/Digg_snapshot_5.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df6=pd.read_csv('data/graphes/digg/Digg_snapshot_6.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df7=pd.read_csv('data/graphes/digg/Digg_snapshot_7.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df8=pd.read_csv('data/graphes/digg/Digg_snapshot_8.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df9=pd.read_csv('data/graphes/digg/Digg_snapshot_9.edgelist.txt', delimiter='\s+', header=None, usecols=[0, 1])
    df=[df1, df2, df3, df4, df5, df6, df7, df8, df9]
    alert_init=0.5
    detect_init=0.7
col1, col2 = st.columns(2)
st.markdown("")
btn1, btn2 = st.columns(2)
#Modify parameters
with btn1:
    with st.popover(":gear: Modifier les paramètres"):
        st.write("""
        :gear: Modifier les paramètres du test 
        """)
        window_size = st.number_input('Introduire la taille de la fenêtre', min_value=1, value=24, placeholder="Taille de la fenêtre")
        metric_input=st.selectbox('Choisir la métrique de détection', ['Wasserstein d\'ordre 1', 'Wasserstein d\'ordre 2', 'Wasserstein régularisé'], index=1)
        cost_input=st.selectbox('Choisir la fonction de coût', ['Euclidienne', 'Euclidienne Standarisée', 'Mahalanobis'], index=1)
        if metric_input == 'Wasserstein d\'ordre 1':
            ot_metric = relio.OTMetric.WASSERSTEIN1
        elif metric_input == 'Wasserstein d\'ordre 2':
            ot_metric = relio.OTMetric.WASSERSTEIN2
        elif metric_input == 'Wasserstein régularisé':
            ot_metric = relio.OTMetric.SINKHORN

        if cost_input == 'Euclidienne':
            cost_function = relio.CostFunction.EUCLIDEAN
        elif cost_input == 'Euclidienne Standarisée':
            cost_function = relio.CostFunction.SEUCLIDEAN
        elif cost_input == 'Mahalanobis':    
            cost_function = relio.CostFunction.MAHALANOBIS
        alert_thold=st.number_input('Introduire le Pourcentage d\'alerte', value=alert_init, placeholder="Pourcentage d'alerte")
        detect_thold=st.number_input('Introduire le Pourcentage de détection', value=detect_init, placeholder="Pourcentage de détection")
        stblty_thold=st.number_input('Introduire le seuil de stabilité', min_value=1, value=3, placeholder="Seuil de stabilité")

api=relio.RELIO_API(24,alert_thold, detect_thold, ot_metric, cost_function, stblty_thold,df, 1)
ref_dist=df[0]
first_concept=relio.Concept(1, np.array(ref_dist))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window=[]
with col1:
    st.markdown(f"""
        :small_red_triangle_down: Taille de la fenêtre : ***{window_size} Heures***
    """, help="c'est le :blue-background[nombre d'heures] à considérer pour le calcul de la métrique de drift.")
    st.markdown(f"""
        :small_red_triangle_down: Métrique de détection : ***{metric_input}***
    """, help="c'est la métrique basée sur le transport optimal pour :blue-background[comparer les distributions] de données afin de détecter le drift.Le transport optimal possède une variété de métriques. Nous avons opté pour celles les plus utilisées dans la littérature.")
    st.markdown(f"""
        :small_red_triangle_down: Fonction de coût : ***{cost_input}***
    """, help=" c'est une :blue-background[distance calculée entre les paires de données] de deux distibutions, utilisée par les métriques du transport optimal.")
with col2:
    st.markdown(f"""
        :small_red_triangle_down: Pourcentage d'alerte : ***{alert_thold}***
                """, help="c'est le :blue-background[pourcentage de changement de distribution] à partir duquel une alerte est déclenchée. Autrement dit, si la metrique de comparaison augmente de 20% alors une alerte est déclenchée.")
    st.markdown(f"""
        :small_red_triangle_down: Pourcentage de détection : ***{detect_thold}***
                """, help="c'est le :blue-background[pourcentage de changement de distribution] à partir duquel le drift est détecté. Autrement dit, si la metrique de comparaison augmente de 50% alors le drift est détecté.")
    st.markdown(f"""
        :small_red_triangle_down: Seuil de stabilité : ***{stblty_thold} fenêtres***
                """, help="C'est :blue-background[le nombre de fenetre] pour dire que les données sont :blue-background[stables sur une distribution], autrement dit : absence de drift.")    
with btn2:
    button=st.button(":arrow_forward: Lancer le test ", type="primary")
if button:
    st.toast("Initialisation de l'API en cours...", icon="⏳")

    st.write(f"""
    ##### 	:chart_with_upwards_trend: Évolution de la distance de {metric_input} entre la distribution de référence et la fenêtre courante  : 
    """)
    distances=st.empty()

    st.divider()

    st.write("""
            ### :clock1: Historique des drifts détectés: 
    """)
    i=0
    for current_window in df[1:]:
        i+=1
        api.set_curr_win(np.array(current_window))
        api.monitorDrift()

        if(api.get_action()==0):
            drift_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.toast(f":red[Un drift est détecté à la fenetre d'indice{i} à {drift_time}]", icon="⚠️")
            st.error(f"Un drift est détecté à la fenetre d'indice{i} à {drift_time}", icon="⚠️")
            drift_type=api.identifyType()
            if(drift_type != None):
                if drift_type == relio.DriftType.GRADUAL:
                    st.toast(f':blue[Le type de drift est : Graduel]', icon="📌")
                    st.info(f'Le type de drift est : Graduel', icon="📌")
                elif drift_type == relio.DriftType.SUDDEN:
                    st.toast(f':blue[Le type de drift est : Soudain]', icon="📌")
                    st.info(f'Le type de drift est : Soudain', icon="📌")
                elif drift_type == relio.DriftType.RECURRENT:
                    st.toast(f':blue[Le type de drift est : Récurrent]', icon="📌")
                    st.info(f'Le type de drift est : Récurrent', icon="📌")
                elif drift_type == relio.DriftType.INCREMENTAL:
                    st.toast(f':blue[Le type de drift est : Incrémental]', icon="📌")
                    st.info(f'Le type de drift est : Incrémental', icon="📌")
            api.reset_retrain_model()

            ref_dist=current_window
        elif (api.get_action()==1):
            alert_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.toast(f"Alerte : Un petit changement de distribution s'est produit  à partir de la fenetre d'indice {i+1-window_size} à {alert_time}!", icon="❗")
            st.warning(f"Alerte : Un petit changement de distribution s'est produit  à partir de la fenetre d'indice {i+1-window_size} à {alert_time}!", icon="❗")
            api.reset_partial_fit()
        distances_data=pd.DataFrame(api.get_distances()[:i], columns=['Distance'])
        distances_data['Alerte']=api.get_alert_thold()
        distances_data['Détection']=api.get_detect_thold()
        distances.line_chart(distances_data, color=["#FFAC1C","#338AFF", "#FF0D0D"])
        drift_type=api.identifyType()
        if(drift_type != None):
            if drift_type == relio.DriftType.GRADUAL:
                st.toast(f':blue[Le type de drift est : Graduel]', icon="📌")
                st.info(f'Le type de drift est : Graduel', icon="📌")
            elif drift_type == relio.DriftType.SUDDEN:
                st.toast(f':blue[Le type de drift est : Soudain]', icon="📌")
                st.info(f'Le type de drift est : Soudain', icon="📌")
            elif drift_type == relio.DriftType.RECURRENT:
                st.toast(f':blue[Le type de drift est : Récurrent]', icon="📌")
                st.info(f'Le type de drift est : Récurrent', icon="📌")
            elif drift_type == relio.DriftType.INCREMENTAL:
                st.toast(f':blue[Le type de drift est : Incrémental]', icon="📌")
                st.info(f'Le type de drift est : Incrémental', icon="📌")