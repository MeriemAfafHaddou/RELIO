import streamlit as st
import pandas as pd
import RELIO_API as relio
import time
import numpy as np
import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from frouros.detectors.data_drift import EMD
from frouros.detectors.data_drift import JS
from frouros.callbacks.batch import PermutationTestDistanceBased
from frouros.datasets.synthetic import SEA

st.logo("images/logo.png")
st.set_page_config(
   page_title="Comparaison",
   page_icon="images/icon.png",
   layout="wide",
   initial_sidebar_state="expanded",
)
st.write("""
# Comparaison entre RELIO et les méthodes classiques
""")
st.markdown("####")
pca = PCA(n_components=1)

# Dataset choice
option = st.selectbox(
    ":bar_chart: Quel dataset voulez vous choisir?",
    ("Simulation : Drift Soudain","Simulation : Drift Graduel","Simulation : Drift Récurrent","Simulation : Drift Incrémental","Synthétique : Insects Soudain","Synthétique : Insects Incrémental","Synthétique : SEA","Asfault", "Electricity","Outdoor Objects", "Ozone"))

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
elif option == "Synthétique : Insects Soudain":
    df=pd.read_csv('data/insects_sudden.csv', header=None)[9800:13800]
    alert_thold=5.4
    detect_thold=6.2
    win_size=500
elif option == "Synthétique : Insects Incrémental":
    df=pd.read_csv('data/insects_incremental.csv', header=None)[32000:37000]
    alert_thold=4.7
    detect_thold=5.3
    win_size=500
elif option == "Synthétique : SEA":
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
    df['class'] = ints
    win_size=50
#Modify parameters
col1, col2 = st.columns(2)
st.markdown("")
btn1, btn2 = st.columns(2)
with btn1:
    with st.popover(":gear: Modifier les paramètres"):
        st.write("""
        :gear: Modifier les paramètres du test 
        """)
        window_size = st.number_input('Introduire la taille de la fenêtre', min_value=1, value=win_size, placeholder="Taille de la fenêtre")
        metric_input=st.selectbox('Choisir la métrique de détection', ['Wasserstein d\'ordre 1', 'Wasserstein d\'ordre 2', 'Wasserstein régularisé'], index=0)
        cost_input=st.selectbox('Choisir la fonction de coût', ['Euclidienne', 'Euclidienne Standarisée', 'Mahalanobis'], index=0)
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
        alert_thold=st.number_input('Introduire le Pourcentage d\'alerte', min_value=1, value=20, placeholder="Pourcentage d'alerte")
        detect_thold=st.number_input('Introduire le Pourcentage de détection', min_value=1, value=50, placeholder="Pourcentage de détection")
        stblty_thold=st.number_input('Introduire le seuil de stabilité', min_value=1, value=3, placeholder="Seuil de stabilité")

pc1 = pca.fit_transform(df.iloc[:,:-1])

#API initialization
api=relio.RELIO_API(window_size, alert_thold, detect_thold, ot_metric, cost_function, stblty_thold, pd.DataFrame(pc1) )


with col1:
    st.markdown(f"""
        :small_red_triangle_down: Taille de la fenêtre : ***{window_size} Données***
    """, help="c'est le :red-background[nombre de données] à considérer pour le calcul de la métrique de drift.")
    st.markdown(f"""
        :small_red_triangle_down: Métrique de détection : ***{metric_input}***
    """, help="c'est la métrique basée sur le transport optimal pour :red-background[comparer les distributions] de données afin de détecter le drift.Le transport optimal possède une variété de métriques. Nous avons opté pour celles les plus utilisées dans la littérature.")
    st.markdown(f"""
        :small_red_triangle_down: Fonction de coût : ***{cost_input}***
    """, help=" c'est une :red-background[distance calculée entre les paires de données] de deux distibutions, utilisée par les métriques du transport optimal.")
with col2:
    st.markdown(f"""
        :small_red_triangle_down: Pourcentage d'alerte : ***{alert_thold}%***
                """, help="c'est le :red-background[pourcentage de changement de distribution] à partir duquel une alerte est déclenchée. Autrement dit, si la metrique de comparaison augmente de 20% alors une alerte est déclenchée.")
    st.markdown(f"""
        :small_red_triangle_down: Pourcentage de détection : ***{detect_thold}%***
                """, help="c'est le :red-background[pourcentage de changement de distribution] à partir duquel le drift est détecté. Autrement dit, si la metrique de comparaison augmente de 50% alors le drift est détecté.")
    st.markdown(f"""
        :small_red_triangle_down: Seuil de stabilité : ***{stblty_thold} fenêtres***
                """, help="C'est :red-background[le nombre de fenetre] pour dire que les données sont :red-background[stables sur une distribution], autrement dit : absence de drift.")    
fr_win=np.array([])
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
alpha = 0.05
fr_ref=pc1[:win_size]
_ = emd_detector.fit(X=fr_ref)
_ = js_detector.fit(X=fr_ref)

ref_dist=[]
for i in range(window_size):
    ref_dist.append(df.iloc[i])
first_concept=relio.Concept(1, np.array(fr_ref))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window=[]
drifts = []
ref_dist_X = fr_ref
ref_dist_y = np.array(ref_dist)[:, -1].astype(int)
all_classes=np.unique(np.array(df)[:,-1].astype(int))

with btn2:
    button=st.button(":arrow_forward: Lancer le test ", type="primary")
if button:
    st.toast("Initialisation de l'API en cours...", icon="⏳")
    print(api.get_alert_thold())
    print(api.get_detect_thold())
    st.write("""
    ##### :bar_chart: Évolution de la distribution de données : 
    """)
    chart = st.empty()
    st.write(f"""
       🔻 Qualité de la présentation de l'axe 1 =  **{pca.explained_variance_ratio_[0] * 100:.2f}%**
    """)
    st.divider()
    st.write("""
            ### :scales: Comparaison des résultats de détection: 
    """)
    relio_col, fr_col1, fr_col2 = st.columns(3)
    with relio_col:
        st.markdown("""##### RELIO """)
    with fr_col1:
        st.markdown("""##### Frouros - EMD """, help="Earth Mover's Distance : ou bien Wasserstein d'ordre 1")
    with fr_col2:
        st.markdown("""##### Frouros - JS """, help="Jensen-Shannon Distance : une distance basée sur la divergence de Kullback-Leibler")

    for i in range(window_size, len(df)+1):
        # Plot the data from the start to the current point
        chart.line_chart(pc1[:i])
        current_window.append(df.iloc[i-1])   
        if len(current_window) == window_size:
            fr_win=pc1[i-window_size:i]
            win_X=fr_win
            api.set_curr_win(np.array(win_X))
            api.monitorDrift()
            
            win_y=np.array(current_window)[:, -1].astype(int)            
            if(api.get_action()==0):
                drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(f":red[Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}]", icon="⚠️")
                with relio_col:
                    st.error(f" RELIO : Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}", icon="⚠️")
                drift_type=api.identifyType()
                with relio_col :
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

                train_X=np.concatenate((ref_dist_X, win_X))
                train_y=np.concatenate((ref_dist_y, win_y))
              
                ref_dist_X=win_X
                ref_dist_y=win_y
            elif (api.get_action()==1):
                alert_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(f"Alerte : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
                with relio_col:
                    st.warning(f"Alerte : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
                train_X=np.concatenate((ref_dist_X, win_X))
                train_y=np.concatenate((ref_dist_y, win_y))               
                api.reset_ajust_model()            
            
            print(f"DISTANCE = {api.get_distances()[-1]}")
            emd_dist,callbacks_log=emd_detector.compare(X=fr_win)
            p_value = callbacks_log["permutation_test"]["p_value"]
            if(p_value <= alpha):
                with fr_col1:
                    drift_time = datetime.datetime.now().strftime("%H:%M:%S")
                    st.error(f" Frouros : Un drift est détecté à partir de la donnée d'indice  {i+1-window_size} à {drift_time}", icon="⚠️")
                    _ = emd_detector.fit(X=fr_win)
            
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
        with relio_col:
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