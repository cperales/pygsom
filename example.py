from sklearn.datasets import make_classification
from gsom import GSOM


X, y = make_classification()

gsom = GSOM(X=X, y=y, spread_factor=0.5)
gsom.train()
