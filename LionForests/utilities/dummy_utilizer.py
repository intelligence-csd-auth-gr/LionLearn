import numpy as np
class DummyUtilizer():
    def __init__(self):
        self.utilizername = "So dymmy"

    def transform(self, X):
        return np.array(X)

    def inverse_transform(self, X):
        return np.array(X)