import streamlit as st
import pandas as pd
import OT2D_API as ot2d
import time
import numpy as np

st.write("""
# Optimal Transport Drift Detection
Hello *world!*
""")
 
df = pd.read_csv("iris_sudden.csv")
chart = st.empty()

window_size=50
alert_thold=0.8
detect_thold=1.2
stblty_thold=4

api=ot2d.OT2D(window_size, alert_thold, detect_thold, ot2d.OTMetric.WASSERSTEIN2, ot2d.CostFunction.SEUCLIDEAN, stblty_thold )
ref_dist=[]
for i in range(window_size):
    ref_dist.append(df.iloc[i])
first_concept=ot2d.Concept(1, np.array(ref_dist))
api.add_concept(first_concept)
api.set_curr_concept(first_concept)
current_window=[]

for i in range(window_size, len(df)+1):
    # Plot the data from the start to the current point
    chart.line_chart(df[['petal_length', 'petal_width']].iloc[:i])
    current_window.append(df.iloc[i-1])
    if len(current_window) == window_size:
        api.set_curr_win(np.array(current_window))
        api.monitorDrift()
        if(api.get_action()==0):
            st.error('A drift is detected')
            drift_type=api.identifyType()
            if(drift_type != None):
                print("Drift type")
                print(drift_type)
            api.reset_retrain_model()
        current_window=[]
    drift_type=api.identifyType()
    if(drift_type != None):
        print("Drift type")
        print(drift_type)
        st.warning('Drift type: '+str(drift_type))
    # Pause for a moment
    time.sleep(0.05)
