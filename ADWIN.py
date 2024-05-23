import numpy as np
from river import drift

class MultiDimADWIN:
    def __init__(self, delta=0.002):
        self.adwin_detectors = []
        self.delta = delta

    def update(self, X):
        if not self.adwin_detectors:
            self.adwin_detectors = [drift.ADWIN(delta=self.delta) for _ in range(X.shape[1])]
        elif len(self.adwin_detectors) != X.shape[1]:
            self.adwin_detectors = [drift.ADWIN(delta=self.delta) for _ in range(X.shape[1])]

        for i, val in enumerate(X):
            self.adwin_detectors[i].update(val)        
            for i, val in enumerate(X):
                self.adwin_detectors[i].update(val)
    
    def drift_detected(self):
        return any(detector.drift_detected for detector in self.adwin_detectors)
