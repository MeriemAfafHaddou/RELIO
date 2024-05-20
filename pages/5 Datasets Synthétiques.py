import streamlit as st
import pandas as pd
import OT2D_API as ot2d
import time
import numpy as np
import datetime
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
st.write("""
# OT2D : Tests sur des Datasets Synthétiques
""")
with st.expander(":blue[:question: Qu'est-ce qu'un dataset synthétique ?]",expanded=False):
    st.write('''
        Un dataset synthétique est un ensemble de données générées artificiellement pour simuler un scénario particulier. Dans notre cas, nous utiliserons des datasets générés par la bibliothèque River pour simuler des scénarios de changement de distribution.
    ''')
with st.expander(":blue[:question: Comment valider notre solution sur ces datasets ?]",expanded=False):
    st.write('''
        Nous allons tester la fiabilité de notre solution sur des datasets synthétiques, en surveillant les performances du modèle supervisé ou non supervisé.
    ''')
st.write("""
         ### Test : 
""")
# Dataset choice
option = st.selectbox(
    ":bar_chart: Quel dataset voulez vous choisir?",
    ("Insects : Soudain", "Insects : Incrémental"))
df = pd.read_csv("data/iris_sudden.csv")

if option == "Insects : Soudain":
    df=pd.read_csv("data/insects_sudden.csv", header=None)[9800:13800]
elif option == "Insects : Incrémental":
    df=pd.read_csv("data/insects_incremental.csv", header=None)[32000:40000]

#Modify parameters
with st.popover(":gear: Modifier les paramètres"):
    st.write("""
     :gear: Modifier les paramètres du test 
     """)
    window_size = st.number_input('Introduire la taille de la fenêtre', min_value=1, value=500, placeholder="Taille de la fenêtre")
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
    alert_thold=st.number_input('Introduire le seuil d\'alerte', min_value=0.1, value=5.4, placeholder="Seuil d'alerte")
    detect_thold=st.number_input('Introduire le seuil de détection', min_value=0.1, value=6.2, placeholder="Seuil de détection")
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

drift_impacts=[]
accuracies=[]

model = SGDClassifier()
drifted_model=SGDClassifier()
ref_dist_X = np.array(ref_dist)[:, :-1]
ref_dist_y = np.array(ref_dist)[:, -1].astype(int)
model.partial_fit(ref_dist_X, ref_dist_y, classes=np.unique(ref_dist_y))
drifted_model=model
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
pc1 = pca.fit_transform(df)
button=st.button(":arrow_forward: Lancer le test ", type="primary")
if button:
    st.toast("Initialisation de l'API en cours...", icon="⏳")

    st.write("""
    ##### :bar_chart: Évolution de la distribution de données : 
    """)
    chart = st.empty()

    st.write(f"""
    ##### 	:chart_with_upwards_trend: Évolution de la distance de {metric_input}  : 
    """)
    distances=st.empty()

    st.write(f"""
    ##### 	📉 Évolution de la précision : 
    """) 
    accuracy_chart=st.empty()

    st.divider()

    st.write("""
            ### :clock1: Historique des drifts détectés: 
    """)
    for i in range(window_size, len(df)+1):
        # Plot the data from the start to the current point
        chart.line_chart(pc1[:i])
        current_window.append(df.iloc[i-1])
        if len(current_window) == window_size:
            api.set_curr_win(np.array(current_window))
            api.monitorDrift()

            win_X=np.array(current_window)[:, :-1]
            win_y=np.array(current_window)[:, -1].astype(int)

            y_pred = model.predict(win_X)
            accuracy = accuracy_score(y_pred, win_y)
            accuracies.append(accuracy)

            y_pred=drifted_model.predict(win_X)
            drifted_accuracy=accuracy_score(y_pred, win_y)
            drift_impacts.append(drifted_accuracy)
            

            distances_data=pd.DataFrame(api.get_distances()[:i], columns=['Distance'])
            distances_data['Alerte']=alert_thold
            distances_data['Détection']=detect_thold
            distances.line_chart(distances_data, color=["#FFAC1C","#338AFF", "#FF0D0D"])

            accuracy_data=pd.DataFrame()
            accuracy_data['Avec adaptation']=accuracies[:i]
            accuracy_data['Sans adaptation']=drift_impacts[:i]
            accuracy_chart.line_chart(accuracy_data, color=["#338AFF", "#FF0D0D"])

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

                param_grid = {
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'max_iter': [1000, 2000, 3000]
                }
                print(f"UNIQUE : {np.unique(win_y)}")
                train_X=np.concatenate((ref_dist_X, win_X))
                train_y=np.concatenate((ref_dist_y, win_y))
                grid_search = GridSearchCV(estimator=SGDClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', error_score='raise')
                grid_search.fit(train_X, train_y)
                
                best_params = grid_search.best_params_
                print("Best hyperparameters found:", best_params)
                
                # Réentraîner le modèle avec les nouveaux hyperparamètres
                model = SGDClassifier(**best_params)
                model.fit(train_X, train_y)
                ref_dist_X=win_X
                ref_dist_y=win_y
            elif (api.get_action()==1):
                st.toast(f"Alerte : Un petit changement de distribution s'est produit !", icon="❗")
                st.warning(f"Alerte : Un petit changement de distribution s'est produit !", icon="❗")
                train_X=np.concatenate((ref_dist_X, win_X))
                train_y=np.concatenate((ref_dist_y, win_y))
                model.partial_fit(train_X, train_y,classes=np.unique(train_y))
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