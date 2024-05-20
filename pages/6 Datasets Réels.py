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
from sklearn.preprocessing import LabelEncoder

pca = PCA(n_components=1)
st.write("""
# OT2D : Tests sur des Datasets Réels
""")

with st.expander(":blue[:question: Comment évaluer notre solution sur un dataset réel ?]",expanded=False):
    st.write('''
        Nous allons tester notre solution sur des datasets réels pour évaluer sa capacité à détecter les différents types de drifts, en surveillant les performances du modèle supervisé ou non supervisé.
    ''')
st.write("""
         ### Test : 
""")
# Dataset choice
option = st.selectbox(
    ":bar_chart: Quel dataset voulez vous choisir?",
    ("Asfault", "Electricity","Outdoor Objects", "Ozone"))

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
    alert_thold=7.6
    detect_thold=8.6
    win_size=200


#Modify parameters
with st.popover(":gear: Modifier les paramètres"):
    st.write("""
     :gear: Modifier les paramètres du test 
     """)
    window_size = st.number_input('Introduire la taille de la fenêtre', min_value=1, value=win_size, placeholder="Taille de la fenêtre")
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

drift_impacts=[]
accuracies=[]
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
pc1 = pca.fit_transform(df)
button=st.button(":arrow_forward: Lancer le test ", type="primary")
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'max_iter': [1000, 2000, 3000]
}
grid_search = GridSearchCV(estimator=SGDClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', error_score='raise')
grid_search.fit(ref_dist_X, ref_dist_y)
best_params = grid_search.best_params_
model = SGDClassifier(**best_params, random_state=42)
model.partial_fit(ref_dist_X, ref_dist_y, all_classes)
drifted_model=SGDClassifier(**best_params,random_state=42)
drifted_model.partial_fit(ref_dist_X, ref_dist_y, all_classes)
if button:
    st.toast("Initialisation de l'API en cours...", icon="⏳")

    st.write("""
    ##### :bar_chart: Évolution de la distribution de données : 
    """)
    chart = st.empty()
    st.write(f"""
       🔻 Qualité de la présentation de l'axe 1 =  **{pca.explained_variance_ratio_[0]:.2f}**
    """)

    st.write(f"""
    ##### 	:chart_with_upwards_trend: Évolution de la distance de {metric_input} entre la distribution de référence et la fenêtre courante  : 
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

            y_pred_drift=drifted_model.predict(win_X)
            drifted_accuracy=accuracy_score(y_pred_drift, win_y)
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

                print(f"UNIQUE : {np.unique(win_y)}")
                train_X=np.concatenate((ref_dist_X, win_X))
                train_y=np.concatenate((ref_dist_y, win_y))
                model.fit(train_X, train_y)
                ref_dist_X=win_X
                ref_dist_y=win_y
            elif (api.get_action()==1):
                alert_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(f"Alerte : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
                st.warning(f"Alerte : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
                model.partial_fit(win_X, win_y)
                api.reset_ajust_model()
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