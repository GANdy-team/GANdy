import deepchem as dc
import numpy as np
import random
import pandas as pd
from pandas import Series, DataFrame
from deepchem.molnet import load_qm9

def generate_analytical_data(to_csv = True):

	x1 = np.random.uniform(0, 1000, 1000)
	x2 = np.random.uniform(0, 1000, 1000)
	mu = 0
	sigma = (x1+x2)/2

	def f(x1, x2):
    		f_data = np.sin(x1)+np.cos(x2)
    		return f_data

	def g(x1, x2):
    		g_data = np.random.normal(mu, np.abs(sigma), 1000)
    		return g_data

	y = f(x1, x2) + g(x1, x2)

	gen_data = pd.DataFrame({'X1':x1, 'X2':x2, 'Y':y})
    
	if to_csv:
		gen_data.to_csv("analytical_data.csv", index=False, sep = ',')

	return gen_data


def generate_qm9_noise_data(x1,x2,y):

    #load data
	qm9_tasks, datasets, transformers = load_qm9()
	train_dataset, valid_dataset, test_dataset = datasets

	c1=qm9_tasks[x1-1]
	c2=qm9_tasks[x2-1]
	c3=qm9_tasks[y-1]
	
    #extrct the 'y'values
	Y = test_dataset.y
	YT = Y.T

	X1 = YT[x1-1]
	X2 = YT[x2-1]
	Y_a = YT[y-1]

	x1 = X1.tolist()
	x2 = X2.tolist()
	y_l = Y_a.tolist()
	l = Y_a.shape

	n = np.random.uniform(0, l, 1).astype(np.int) #set the number of noise added
	ni = np.random.uniform(0, l, n) #n random values
	an = len(ni)


    #add noise to n numbers of y
	for i in range(an):
		mu = 0
		sigma = (x1[i]+x2[i])/2
		noise = np.random.normal(mu, np.abs(sigma), n)
		g = noise.tolist()
		y_l[i] += g[i]

    #save to_csv:
	dataframe = pd.DataFrame({'x1_'+c1:X1,'x2_'+c2:X2,'y_'+c3:y_l})
	dataframe.to_csv("qm9_noise_data.csv", index=False, sep = ',')
	gen_data = pd.read_csv('qm9_noise_data.csv')

	return gen_data
