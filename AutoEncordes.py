# Dimensionality Reduction with Linear autoencoders

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=100, n_features=3,centers=2, centers=2,random_state=101)


###########################
#recommendation based on net sale qty

sale_qty = sales.groupby(['Product Code'])['Net Sales Qty'].sum()
sale_qty.sort_values('Net Sales Qty' ascending=False).head()

