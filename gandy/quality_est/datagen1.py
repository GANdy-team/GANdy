
import numpy as np
import pandas as pd
import random
import deepchem as dc
from pandas import Series, DataFrame
from deepchem.molnet import load_qm9


def create_data(to_csv = True):

	#import qm9 test data
	qm9_tasks, datasets, transformers = load_qm9()
	train_dataset, valid_dataset, test_dataset = datasets


	Y = test_dataset.y
	YT = Y.T

	#set x1 equals y_mu subset
	y_mu = YT[0]
	x1 = y_mu.tolist()
	

	#set x2 equals y_zpve subset
	y_zpve = YT[6]
	x2 = y_zpve.tolist()
	
	# set y equals y_gap subset
	y_gap = YT[4]
	y = y_gap.tolist()


	#length of y
	l = len(y_gap)
	
	
	
	n = np.random.uniform(0, l, 1).astype(np.int) #set the number of noise added
	ni = np.random.uniform(0, l, n) #n random values
	a = len(ni)
	
	
	#add noise to n numbers of y
	for i in range(a):
    		mu = 0
    		sigma = (x1[i]+x2[i])/2
    		noise = np.random.normal(mu, np.abs(sigma), n)
    		g = noise.tolist()
    		y[i] += g[i]


	
	gen_data = pd.DataFrame({'Y_gap':y, 'X1_mu':x1, 'X2_zpve':x2})


	if to_csv:
		gen_data.to_csv("test_data_qm9.csv", index=False, sep = ',')

	return gen_data
