import streamlit as st
import pandas as pd
import OT2D_API as ot2d
import time
import numpy as np
import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from frouros.detectors.data_drift import EMD
from frouros.detectors.data_drift import JS
from frouros.callbacks.batch import PermutationTestDistanceBased

st.set_page_config(layout="wide")
st.write("""
# Comparaison entre OT2D et les méthodes classiques
""")
with st.expander(":blue[:question: Quelles sont les méthodes classiques de détection du concept drift ?]",expanded=False):
    st.write('''
    ### 1. ADWIN : une méthode de détection de dérive de concept adaptative qui ajuste automatiquement la taille de la fenêtre de données en fonction de la distribution des données.
    ### 2. Page-Hinkley : une méthode de détection de dérive de concept basée sur la somme cumulée des différences entre les valeurs observées et les valeurs attendues.
    ''')
st.write("""
         ### Comparaison : 
""")
pca = PCA(n_components=1)

# Dataset choice
option = st.selectbox(
    ":bar_chart: Quel dataset voulez vous choisir?",
    ("Simulation : Drift Soudain","Simulation : Drift Graduel","Simulation : Drift Récurrent","Simulation : Drift Incrémental","Insects : Soudain","Insects : Incrémental","Asfault", "Electricity","Outdoor Objects", "Ozone"))

if option == "Asfault":
    df=pd.read_csv("data/Asfault.csv", header=None)[:5000]
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df[64])
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df = df.drop(64,axis='columns')
    df.columns= df.columns.astype(str)
    alert_thold=6.5
    detect_thold=7.0
    win_size=250

elif option == "Electricity":
    df=pd.read_csv('data/electricity.csv')[:8000]
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['class'])
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df = df.drop('class',axis='columns')
    df.columns= df.columns.astype(str)
    alert_thold=2.2
    detect_thold=2.3
    win_size=500

elif option == "Outdoor Objects":
    df=pd.read_csv("data/outdoor.csv", header=None)
    alert_thold=4.5
    detect_thold=5.0
    win_size=500
  
elif option == "Ozone":
    df=pd.read_csv('data/Ozone.csv', header=None)
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df[72])
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    df = df.drop(72,axis='columns')
    df.columns= df.columns.astype(str)
    alert_thold=6.0
    detect_thold=7.0
    win_size=200
elif option == "Simulation : Drift Soudain":
    df=pd.read_csv('data/iris_sudden.csv')
    alert_thold=0.9
    detect_thold=1.2
    win_size=50
elif option == "Simulation : Drift Graduel":
    df=pd.read_csv('data/iris_graduel.csv')
    alert_thold=1.5
    detect_thold=1.7
    win_size=20
elif option == "Simulation : Drift Récurrent":
    df=pd.read_csv('data/iris_recurrent.csv')
    alert_thold=1.5
    detect_thold=1.7
    win_size=20
elif option == "Simulation : Drift Incrémental":
    df=pd.read_csv('data/iris_incremental.csv')
    alert_thold=1.65
    detect_thold=2.0
    win_size=40
elif option == "Insects : Soudain":
    df=pd.read_csv('data/insects_sudden.csv', header=None)[9800:13800]
    alert_thold=5.4
    detect_thold=6.2
    win_size=500
elif option == "Insects : Incrémental":
    df=pd.read_csv('data/insects_incremental.csv', header=None)[32000:37000]
    alert_thold=4.7
    detect_thold=5.3
    win_size=500
#Modify parameters
with st.popover(":gear: Modifier les paramètres"):
    st.write("""
     :gear: Modifier les paramètres du test 
     """)
    window_size = st.number_input('Introduire la taille de la fenêtre', min_value=1, value=win_size, placeholder="Taille de la fenêtre")
    metric_input=st.selectbox('Choisir la métrique de détection', ['Wasserstein d\'ordre 1', 'Wasserstein d\'ordre 2', 'Wasserstein régularisé'], index=0)
    cost_input=st.selectbox('Choisir la fonction de coût', ['Euclidienne', 'Euclidienne Standarisée', 'Mahalanobis'], index=0)
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
    alert_thold=st.number_input('Introduire le seuil d\'alerte', min_value=0.1, value=alert_thold, placeholder="Seuil d'alerte")
    detect_thold=st.number_input('Introduire le seuil de détection', min_value=0.1, value=detect_thold, placeholder="Seuil de détection")
    stblty_thold=st.number_input('Introduire le seuil de stabilité', min_value=1, value=3, placeholder="Seuil de stabilité")

#API initialization
api=ot2d.OT2D(window_size, alert_thold, detect_thold, ot_metric, cost_function, stblty_thold )
ref_dist=[]
for i in range(window_size):
    ref_dist.append(df.iloc[i])
first_concept=ot2d.Concept(1, np.array(ref_dist))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window=[]
drifts = []
ref_dist_X = np.array(ref_dist)[:, :-1]
ref_dist_y = np.array(ref_dist)[:, -1].astype(int)
all_classes=np.unique(np.array(df)[:,-1].astype(int))

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
pc1 = pca.fit_transform(df.iloc[:,:-1])
fr_win=np.array([])
emd_detector = EMD()
js_detector=JS(
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
alpha = 0.01
fr_ref=pc1[:win_size]
_ = emd_detector.fit(X=fr_ref)
_ = js_detector.fit(X=fr_ref)
button=st.button(":arrow_forward: Lancer le test ", type="primary")
if button:
    st.toast("Initialisation de l'API en cours...", icon="⏳")

    st.write("""
    ##### :bar_chart: Évolution de la distribution de données : 
    """)
    chart = st.empty()
    st.write(f"""
       🔻 Qualité de la présentation de l'axe 1 =  **{pca.explained_variance_ratio_[0]:.2f}**
    """)
    st.divider()
    st.write("""
            ### :scales: Comparaison des résultats de détection: 
    """)
    ot2d_col, fr_col1, fr_col2 = st.columns(3)
    with ot2d_col:
        st.write("""##### OT2D """)
    with fr_col1:
        st.write("""##### Frouros - EMD """)
    with fr_col2:
        st.write("""##### Frouros - Jensen Shannon """)

    for i in range(window_size, len(df)+1):
        # Plot the data from the start to the current point
        chart.line_chart(pc1[:i])
        current_window.append(df.iloc[i-1])   
        if len(current_window) == window_size:
            api.set_curr_win(np.array(current_window))
            api.monitorDrift()
            fr_win=pc1[i-window_size:i]
            win_X=np.array(current_window)[:, :-1]
            win_y=np.array(current_window)[:, -1].astype(int)            
            if(api.get_action()==0):
                drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(f":red[Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}]", icon="⚠️")
                with ot2d_col:
                    st.error(f" OT2D : Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}", icon="⚠️")
                drift_type=api.identifyType()
                with ot2d_col :
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

                train_X=np.concatenate((ref_dist_X, win_X))
                train_y=np.concatenate((ref_dist_y, win_y))
              
                ref_dist_X=win_X
                ref_dist_y=win_y
            elif (api.get_action()==1):
                alert_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(f"Alerte : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
                with ot2d_col:
                    st.warning(f"Alerte : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
                train_X=np.concatenate((ref_dist_X, win_X))
                train_y=np.concatenate((ref_dist_y, win_y))               
                api.reset_ajust_model()            
            
            emd_dist=emd_detector.compare(X=fr_win)[0]
            if(emd_dist[0]>detect_thold):
                with fr_col1:
                    drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                    st.error(f" Frouros : Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}", icon="⚠️")
                    _ = emd_detector.fit(X=fr_win)

            elif(emd_dist[0]>alert_thold):
                with fr_col1:
                    alert_time = datetime.datetime.now().strftime("%H:%M:%S")
                    st.warning(f"Frouros : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
            
            js_dist,callbacks_log=js_detector.compare(X=fr_win)
            p_value = callbacks_log["permutation_test"]["p_value"]
            if(p_value <= alpha):
                with fr_col2:
                    drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                    st.error(f" Frouros : Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}", icon="⚠️")
                    _ = js_detector.fit(X=fr_win)

            current_window=[]
            fr_win=np.array([])


        drift_type=api.identifyType()
        with ot2d_col:
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