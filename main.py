import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import SGDRegressor
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
np.random.seed(1)
NN = 2

def f(x):
    """The simple function to predict."""
    return x * np.sin(x)

# ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 100)).T
#x = np.atleast_2d([1.5]).T

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

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

		abs_err = np.abs(pred_neir - y[ind])  #((pred_neir - y[ind]) ** 2) #
		#print dist
		weights = 1 - (dist / dist.sum())
		#print "x",x[i],"weights",weights
		#print x[i], abs_err, weights, y[ind], pred_neir
		weighted_err = np.average(abs_err, weights=weights**NN) 


		#1/NN sum ( y_i - fx) ^2
		#
		#nbrs_var = 1.0/NN * np.sum( (y[ind] - ) )

		#print weighted_err
		nbrs_y = list(y[ind])
		nbrs_y.append(pred[i])
		nbrs_var = np.std(nbrs_y)

		min_dist = np.min(dist)
		pred_var = weighted_err + min_dist * nbrs_var
		sigma.append(pred_var)
	sigma = np.array(sigma)
	return sigma

sigma_GP = PredVar(nbrs,gp,y_pred,x,y)

## RF now train another model that normally does not provide the prediction variance
RF = RandomForestRegressor(n_estimators=50)
RF.fit(X,y)
y_pred_RF = RF.predict(x)
sigma_RF = PredVar(nbrs,RF,y_pred_RF,x,y)
#now calculate the approximate prediction variance using NN


## kNN regression
knn = neighbors.KNeighborsRegressor(2, weights='distance')
knn.fit(X, y)
y_pred_kNN = knn.predict(x)
sigma_kNN = PredVar(nbrs,knn, y_pred_kNN, x, y)

## SVM
SVM = svm.SVR(C=2, epsilon=0.01)
SVM.fit(X,y)
y_pred_SVM = SVM.predict(x)
sigma_SVM = PredVar(nbrs,SVM,y_pred_SVM,x,y)
#now calculate the approximate prediction variance using NN

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure(figsize=[10,6])
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction Kriging')
plt.fill(np.concatenate([x, x[::-1]]),
          np.concatenate([y_pred - 1.9600 * sigma,
                         (y_pred + 1.9600 * sigma)[::-1]]),
          alpha=.2, fc='b', ec='None', label='Kriging variance')
plt.fill(np.concatenate([x, x[::-1]]),
          np.concatenate([y_pred - sigma_GP,
                         (y_pred + sigma_GP)[::-1]]),
          alpha=.2, fc='g', ec='None', label='Heuristic error Kriging')

#plt.plot(x, y_pred_RF, 'r-', label=u'Prediction RF')
#plt.fill(np.concatenate([x, x[::-1]]),
#         np.concatenate([y_pred_RF - 1.9600 * sigma_RF,
#                        (y_pred_RF + 1.9600 * sigma_RF)[::-1]]),
#         alpha=.2, fc='r', ec='None', label='95% confidence interval RF')

plt.plot(x, y_pred_SVM, 'y-', label=u'Prediction SVR')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred_SVM -  sigma_SVM,
                        (y_pred_SVM +  sigma_SVM)[::-1]]),
         alpha=.2, fc='y', ec='None', label='Heuristic error SVR')

#plt.plot(x, y_pred_kNN, 'r-', label=u'Prediction k-NN')
#plt.fill(np.concatenate([x, x[::-1]]),
#         np.concatenate([y_pred_kNN -  sigma_kNN,
#                        (y_pred_kNN +  sigma_kNN)[::-1]]),
#         alpha=.2, fc='r', ec='None', label='Heuristic prediction error k-NN')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.xlim(0,10)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("example7.png")


