import streamlit as st
import pandas as pd
import OT2D_API as ot2d
import time
import numpy as np
import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

pca = PCA(n_components=1)

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
    model_type=st.selectbox('Choisir le type de modèle', ["Supervisé - Stochastic Gradient Descent", "Non supervisé - KMeans"])
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
drift_impacts=[]
adapt_perform=[]
current_model=0
ref_dist_X = np.array(ref_dist)[:, :-1]
ref_dist_y = np.array(ref_dist)[:, -1].astype(int)
all_classes=np.unique(np.array(df)[:,-1].astype(int))
st.write(f"""
    :small_red_triangle_down: Type de modèle : ***{model_type}***
""")

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
button=st.button(":arrow_forward: Lancer la simulation", type="primary")
if model_type== "Supervisé - Stochastic Gradient Descent":
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
    metric_name="de la Précision"
elif model_type == "Non supervisé - KMeans":
    silhouette_avg = []
    K = range(2, 11)  # Nombre de clusters à tester de 2 à 10 (car silhouette_score n'est pas défini pour k=1)
    for k in K:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(ref_dist_X)
        silhouette_avg.append(silhouette_score(ref_dist_X, cluster_labels))
    n=np.argmax(silhouette_avg)+2
    model= MiniBatchKMeans(n_clusters=n, random_state=42)
    drifted_model=MiniBatchKMeans(n_clusters=n, random_state=42)
    model=model.fit(ref_dist_X)
    cluster_labels_model=model.labels_
    drifted_model=drifted_model.fit(ref_dist_X)
    cluster_labels_drift=drifted_model.labels_
    adapt_perform.append(silhouette_score(ref_dist_X, cluster_labels_model))
    drift_impacts.append(silhouette_score(ref_dist_X,cluster_labels_drift))
    metric_name="du Score Silhouette"
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
    st.divider()
    st.write(f"""
    ##### 	📉 Impact du drift récurrent - Évolution {metric_name}: 
    """) 
    metric_chart=st.empty()

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

            if model_type== "Supervisé - Stochastic Gradient Descent":
                y_pred = model.predict(win_X)
                metric = accuracy_score(y_pred, win_y)

                y_pred_drift=drifted_model.predict(win_X)
                drifted_metric=accuracy_score(y_pred_drift, win_y)
 

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
                train_X=np.concatenate((ref_dist_X, win_X))
                train_y=np.concatenate((ref_dist_y, win_y))
                if model_type== "Supervisé - Stochastic Gradient Descent":
                    model.fit(train_X, train_y)
                elif model_type == "Non supervisé - KMeans":
                    if current_model == 0:
                        silhouette_avg = []
                        K = range(2, 11)  # Nombre de clusters à tester de 2 à 10 (car silhouette_score n'est pas défini pour k=1)
                        for k in K:
                            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
                            cluster_labels = kmeans.fit_predict(ref_dist_X)
                            silhouette_avg.append(silhouette_score(ref_dist_X, cluster_labels))
                        n=np.argmax(silhouette_avg)+2
                        new_model= MiniBatchKMeans(n_clusters=n, random_state=42)
                        labels = new_model.fit(train_X)
                        current_model=1
                    else:
                        labels = model.fit(win_X)      
                        current_model=0        
                ref_dist_X=win_X
                ref_dist_y=win_y

            elif (api.get_action()==1):
                alert_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.toast(f"Alerte : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
                st.warning(f"Alerte : Un petit changement de distribution s'est produit  à partir de la donnée d'indice {i+1-window_size} à {alert_time}!", icon="❗")
                if model_type== "Supervisé - Stochastic Gradient Descent":
                    model.partial_fit(train_X, train_y)
                elif model_type == "Non supervisé - KMeans":
                    model.partial_fit(train_X)                
                api.reset_ajust_model()

            distances_data=pd.DataFrame(api.get_distances()[:i], columns=['Distance'])
            distances_data['Alerte']=alert_thold
            distances_data['Détection']=detect_thold
            distances.line_chart(distances_data, color=["#FFAC1C","#338AFF", "#FF0D0D"])
            
            if model_type == "Non supervisé - KMeans":
                if current_model==0:
                    labels = model.predict(win_X)
                    print("EX model")
                else: 
                    print("new model") 
                    labels = new_model.predict(win_X)
                labels_drift=drifted_model.predict(win_X)
                metric = silhouette_score(win_X, labels)  
                drifted_metric=silhouette_score(win_X, labels_drift)   
            adapt_perform.append(metric)
            drift_impacts.append(drifted_metric)  

            metric_data=pd.DataFrame()
            metric_data['Avec adaptation']=adapt_perform[:i]
            metric_data['Sans adaptation']=drift_impacts[:i]
            metric_chart.line_chart(metric_data, color=["#338AFF", "#FF0D0D"])


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