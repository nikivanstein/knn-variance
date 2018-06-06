import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import SGDRegressor
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from deap import benchmarks
np.random.seed(1)
NN = 50

##
#sklearn.datasets.make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)[source]

bmark = benchmarks.ackley

def f(x):
    """The simple function to predict."""
    y = []
    if (len(x)>1):
    	for x_i in x:
    		y.append(bmark(x_i)[0])
    	return np.array(y)
    else:
    	return bmark(x)[0]
    #return x * np.sin(x)



# ----------------------------------------------------------------------
#  First the noiseless case
X = 30*np.random.rand(100,5)-15
X = X[X[:,0].argsort()]
# Observations
y = f(X)


# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
#x = 30*np.random.rand(100,5)-15
x = np.ones((100,5))
x[:,0] = x[:,0] + np.linspace(-15,15,100)
#x.sort(axis=0)
#x = x[x[:,0].argsort()]
#x = np.atleast_2d([1.5]).T

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

#train nn network
no = MinMaxScaler(copy=True)
nbrs = NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(no.fit_transform(X))


def PredVar(nbrs,model,pred,x,y):
	#Expecte squared error = Expected Bias ^2 + Expected Variance + noise
	#Expected Bias is the Expected error between f^(x) and f(x)
	#We can approximate this error by looking at the prediction error of the known data points
	#and interpolating this error to nearby predictions
	#
	#The expected variance 

	normx = no.transform(x)
	sigma = []
	distances, indices = nbrs.kneighbors(normx,NN)
	
	for i in range(len(x)):
		#dist, ind = nbrs.kneighbors(x[i].reshape(1, -1),NN)
		dist = distances[i]
		ind = indices[i]
		#calculate the neirest point error
		pred_neir = pred[i]

		abs_err = np.abs(pred_neir - y[ind])
		#print dist
		weights = 1 - (dist / dist.sum())
		#print "x",x[i],"weights",weights
		#print x[i], abs_err, weights, y[ind], pred_neir
		weighted_err = np.average(abs_err, weights=weights**NN)
		#print weighted_err
		nbrs_y = list(y[ind])
		nbrs_y.append(pred[i])
		nbrs_var = np.var(nbrs_y)
		min_dist = np.min(dist)
		pred_var = weighted_err + min_dist * nbrs_var
		sigma.append(pred_var)
	sigma = np.array(sigma)
	return sigma

sigma_GP = PredVar(nbrs,gp,y_pred,x,y)

#now train another model that normally does not provide the prediction variance
RF = RandomForestRegressor(n_estimators=100)
RF.fit(X,y)
y_pred_RF = RF.predict(x)
sigma_RF = PredVar(nbrs,RF,y_pred_RF,x,y)

sigma_RF_bootstrap = np.std([tree.predict(x) for tree in RF.estimators_],
             axis=0)
#print sigma_RF_bootstrap

#now calculate the approximate prediction variance using NN

#SVM
SVM = svm.SVR()
SVM.fit(X,y)
y_pred_SVM = SVM.predict(x)
sigma_SVM = PredVar(nbrs,SVM,y_pred_SVM,x,y)
#now calculate the approximate prediction variance using NN

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure(figsize=[7,6])
plt.plot(x[:,0], f(x), 'g.', label=u'$f(x) = ackley$')
#plt.plot(X[:,0], y, 'r.', markersize=10, label=u'Observations')
plt.plot(x[:,0], y_pred, 'b-', label=u'Prediction Kriging')
plt.fill(np.concatenate([x[:,0], x[:,0][::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.2, fc='b', ec='None', label='Kriging var.')
plt.fill(np.concatenate([x[:,0], x[:,0][::-1]]),
         np.concatenate([y_pred - sigma_GP,
                        (y_pred + sigma_GP)[::-1]]),
         alpha=.2, fc='g', ec='None', label='k-NN uncertainty')
plt.xlim([-15,15])
#plt.plot(range(len(x)), y_pred_RF, 'r-', label=u'Prediction RF')
#plt.fill(np.concatenate([range(len(x)), range(len(x))[::-1]]),
#         np.concatenate([y_pred_RF - 1.9600 * sigma_RF,
#                        (y_pred_RF + 1.9600 * sigma_RF)[::-1]]),
#         alpha=.2, fc='r', ec='None', label='95% confidence interval RF')



#plt.fill(np.concatenate([range(len(x)), range(len(x))[::-1]]),
#         np.concatenate([y_pred_RF - 1.9600 * sigma_RF_bootstrap,
#                        (y_pred_RF + 1.9600 * sigma_RF_bootstrap)[::-1]]),
#         alpha=.2, fc='b', ec='None', label='95% confidence interval RFb')

#plt.plot(range(len(x)), y_pred_SVM, 'y-', label=u'Prediction SVR')
#plt.fill(np.concatenate([range(len(x)), range(len(x))[::-1]]),
#         np.concatenate([y_pred_SVM -  sigma_SVM,
#                        (y_pred_SVM +  sigma_SVM)[::-1]]),
#         alpha=.2, fc='y', ec='None', label='95% confidence interval SVR')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
#plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("predvar_GP_ackley_slice_5d.png")


