from frouros.detectors.data_drift import EMD
import numpy as np
np.random.seed(seed=31)
X = np.random.normal(loc=0, scale=1, size=100)
Y = np.random.normal(loc=1, scale=1, size=100)
detector = EMD()
_ = detector.fit(X=X)
distance=detector.compare(X=Y)[0]
print(distance[0])