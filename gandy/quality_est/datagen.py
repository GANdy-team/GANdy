def create_data(to_csv = True):

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

	if to_csv:
		dataframe = pd.DataFrame({'Y':y, 'X1':x1, 'X2':x2})
		dataframe.to_csv("test_data.csv", index=False, sep = ',')

	return x1, x2, y
