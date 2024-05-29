import streamlit as st

st.write(""" # 📑 Documentation de RELIO API """)
ot_container = st.container(border=True)
ot_container.write("""
### :question: Qu'est-ce que le transport optimal?
Le Transport Optimal est une méthode mathématique qui permet de trouver la meilleure correspondance entre deux distributions de probabilité. Il offre des métriques fiable pour comparer les distributions de données.
                  """)

param_container = st.container(border=True)
param_container.write("""
#### :gear: Paramètres de la simulation """)
param_container.markdown("""
    :small_red_triangle_down: **Taille de la fenetre** : c'est le :red-background[nombre de données] à considérer pour le calcul de la métrique de drift. \n
    :small_red_triangle_down: **Métrique de la détection** : c'est la métrique basée sur le transport optimal pour :red-background[comparer les distributions] de données afin de détecter le drift, en :red-background[quantifiant le cout minimal pour transporter une distribution à une autre]. \n
    :small_red_triangle_down: **Fonction de coût** : c'est une :red-background[distance calculée entre les paires de données] de deux distibutions, utilisée par les métriques du transport optimal. On trouve : la distance euclidienne et sa version standarisée, et la distance de Mahalanobis. \n
    :small_red_triangle_down: **Pourcentage d'alerte** : c'est le :red-background[pourcentage de changement de distribution] à partir duquel une alerte est déclenchée. Autrement dit, si la metrique de comparaison augmente de 20% alors une alerte est déclenchée. \n
    :small_red_triangle_down: **Pourcentage de détection** : c'est le :red-background[pourcentage de changement de distribution] à partir duquel le drift est détecté. Autrement dit, si la metrique de comparaison augmente de 50% alors le drift est détecté. \n
    :small_red_triangle_down: **Seuil de stabilité** : C'est :red-background[le nombre de fenetre] pour dire que les données sont :red-background[stables sur une distribution], autrement dit : absence de drift \n
""")