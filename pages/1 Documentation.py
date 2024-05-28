import streamlit as st

st.write(""" # 📑 Documentation de OT2D API """)

st.write("""
#### Paramètres de la simulation """)

st.markdown("""
    * **Taille de la fenetre** : c'est le :blue-background[nombre de données] à considérer pour le calcul de la métrique de drift. \n
    * **Métrique de la détection** : c'est la métrique basée sur le transport optimal pour :blue-background[comparer les distributions] de données afin de détecter le drift.Le transport optimal possède une variété de métriques. Nous avons opté pour celles les plus utilisées dans la littérature. \n
    * **Fonction de coût** : c'est une :blue-background[distance calculée entre les paires de données] de deux distibutions, utilisée par les métriques du transport optimal. On trouve : la distance euclidienne et sa version standarisée, et la distance de Mahalanobis. \n
    * **Pourcentage d'alerte** : c'est le :blue-background[pourcentage de changement de distribution] à partir duquel une alerte est déclenchée. Autrement dit, si la metrique de comparaison augmente de 20% alors une alerte est déclenchée. \n
    * **Pourcentage de détection** : c'est le :blue-background[pourcentage de changement de distribution] à partir duquel le drift est détecté. Autrement dit, si la metrique de comparaison augmente de 50% alors le drift est détecté. \n
    * **Seuil de stabilité** : C'est :blue-background[le nombre de fenetre] pour dire que les données sont :blue-background[stables sur une distribution], autrement dit : absence de drift \n
""")