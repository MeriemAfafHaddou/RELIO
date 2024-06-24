import streamlit as st
import pandas as pd
import RELIO_API as relio
import time
import numpy as np
import datetime
import altair as alt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.logo("images/logo.png")
st.set_page_config(
   page_title="Simulation - Drift Graduel",
   page_icon="images/icon.png",
   layout="wide",
   initial_sidebar_state="expanded",
)
pca = PCA(n_components=1)


st.write("""
# RELIO : Simulation d'un drift graduel
""")
with st.expander(":blue[:question: Qu'est-ce qu'un drift graduel ?]",expanded=False):
    st.write('''
        Il fait référence à un changement progressif où deux sources, Si et Sj,sont actives simultanément pendant un certain temps. Au fil du temps, la probabilité d’arrivée
d’une instance de la source Si diminue, tandis que la probabilité d’arrivée d’une instance de
la source Sj augmente, jusqu’à ce que Sj soit complètement remplacée par Si comme illustré
dans la figure : 
             
    ''')
    st.image('images/graduel.png')

st.write("""
         ### Simulation : 
""")
df = pd.read_csv("data/iris_graduel.csv")
col1, col2 = st.columns(2)
st.markdown("")
btn1, btn2 = st.columns(2)

with btn1:
    with st.popover(":gear: Modifier les paramètres"):
        st.write("""
        :gear: Modifier les paramètres de la simulation 
        """)
        model_type=st.selectbox('Choisir le type de modèle', ["Supervisé - Stochastic Gradient Descent", "Non supervisé - KMeans"])
        window_size = st.number_input('Introduire la taille de la fenêtre', min_value=1, value=20, placeholder="Taille de la fenêtre")
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
        alert_thold=st.number_input('Introduire le pourcentage d\'alerte', min_value=1, value=20, placeholder="Pourcentage d'alerte", step=1)
        detect_thold=st.number_input('Introduire le pourcentage de détection', min_value=1, value=40, placeholder="Pourcentage de détection",step=1)
        stblty_thold=st.number_input('Introduire le seuil de stabilité', min_value=1, value=3, placeholder="Seuil de stabilité",step=1)

api=relio.RELIO_API(window_size, alert_thold, detect_thold, ot_metric, cost_function, stblty_thold, df, 0)

ref_dist=[]
for i in range(window_size):
    ref_dist.append(df.iloc[i])
first_concept=relio.Concept(1, np.array(ref_dist))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window=[]
drift_impacts=[]
adapt_perform=[]
current_model=0
ref_dist_X = np.array(ref_dist)[:, :-1]
ref_dist_y = np.array(ref_dist)[:, -1].astype(int)
all_classes=np.unique(np.array(df)[:,-1].astype(int))

with col1:
    st.markdown(f"""
        :small_red_triangle_down: Taille de la fenêtre : ***{window_size} Données***
    """, help="c'est le :blue-background[nombre de données] à considérer pour le calcul de la métrique de drift.")
    st.markdown(f"""
        :small_red_triangle_down: Métrique de détection : ***{metric_input}***
    """, help="c'est la métrique basée sur le transport optimal pour :blue-background[comparer les distributions] de données afin de détecter le drift.Le transport optimal possède une variété de métriques. Nous avons opté pour celles les plus utilisées dans la littérature.")
    st.markdown(f"""
        :small_red_triangle_down: Fonction de coût : ***{cost_input}***
    """, help=" c'est une :blue-background[distance calculée entre les paires de données] de deux distibutions, utilisée par les métriques du transport optimal.")
    st.markdown(f"""
        :small_red_triangle_down: Type de modèle : ***{model_type}***
    """, help="Pour spécifier si le modèle utilisé est :blue-background[supervisé ou non supervisé].")
with col2:
    st.markdown(f"""
        :small_red_triangle_down: Pourcentage d'alerte : ***{alert_thold}%***
                """, help="c'est le :blue-background[pourcentage de changement de distribution] à partir duquel une alerte est déclenchée. Autrement dit, si la metrique de comparaison augmente de 20% alors une alerte est déclenchée.")
    st.markdown(f"""
        :small_red_triangle_down: Pourcentage de détection : ***{detect_thold}%***
                """, help="c'est le :blue-background[pourcentage de changement de distribution] à partir duquel le drift est détecté. Autrement dit, si la metrique de comparaison augmente de 50% alors le drift est détecté.")
    st.markdown(f"""
        :small_red_triangle_down: Seuil de stabilité : ***{stblty_thold} fenêtres***
                """, help="C'est :blue-background[le nombre de fenetre] pour dire que les données sont :blue-background[stables sur une distribution], autrement dit : absence de drift.")    
pc1 = pca.fit_transform(df.iloc[:,:-1])
with btn2:
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
       🔻 Qualité de la présentation de l'axe 1 =  **{pca.explained_variance_ratio_[0] * 100:.2f}%**
    """)
    st.write(f"""
    ##### 	:chart_with_upwards_trend: Évolution de la distance de {metric_input} entre la distribution de référence et la fenêtre courante  : 
    """)
    distances=st.empty()
    st.divider()
    st.write(f"""
    ##### 	📉 Impact du drift graduel - Évolution {metric_name} : 
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
                    model.partial_fit(win_X, win_y)
 

                elif model_type == "Non supervisé - KMeans":
                    model.partial_fit(train_X)                
                api.reset_partial_fit()
            distances_data=pd.DataFrame(api.get_distances()[:i], columns=['Distance'])
            distances_data['Alerte']=api.get_alert_thold()
            distances_data['Détection']=api.get_detect_thold()
            distances.line_chart(distances_data, color=["#FFAC1C","#338AFF", "#FF0D0D"])
            
            if model_type == "Non supervisé - KMeans":
                if current_model==0:
                    labels = model.predict(win_X)
                else: 
                    labels = new_model.predict(win_X)
                labels_drift=drifted_model.predict(win_X)
                metric = silhouette_score(win_X, labels)  
                drifted_metric=silhouette_score(win_X, labels_drift)   
            adapt_perform.append(metric)
            drift_impacts.append(drifted_metric)  

            metric_data=pd.DataFrame()
            metric_data['Avec adaptation']=adapt_perform[:i]
            metric_data['Impact du drift']=drift_impacts[:i]
            metric_chart.line_chart(metric_data, color=["#338AFF", "#FF0D0D"])


            current_window=[]
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
        # Pause for a moment
        time.sleep(0.1)


if len(adapt_perform) > 0:
    print(f"Drift impact mean: {sum(drift_impacts) / len(drift_impacts)}")
    print(f"Adaptation performance mean: {sum(adapt_perform) / len(adapt_perform)}")