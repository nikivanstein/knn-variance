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

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import math

np.random.seed(1)
NN = 10

##
#sklearn.datasets.make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)[source]

bmark = benchmarks.schaffer
benchmark_name = "schaffer"

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
X = 200*np.random.rand(100,2)-100
# Observations
y = f(X)


# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
# 
delta = 1.0
mesh_X = np.arange(-100.0, 100.0, delta)
mesh_Y = np.arange(-100.0, 100.0, delta)
mesh_X, mesh_Y = np.meshgrid(mesh_X, mesh_Y)
x_0 = mesh_X.flatten()
x_1 = mesh_Y.flatten()

x = []
for i in range(len(x_0)):
	x.append([x_0[i],x_1[i]])
x = np.array(x)


#x = np.atleast_2d([1.5]).T

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-5, 1e5)) * RBF(10, (1e-5, 1e5))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

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

#ANN
from sklearn.neural_network import MLPRegressor
ANN = MLPRegressor(hidden_layer_sizes=(100,50 ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ANN.fit(X,y)
y_pred_ANN = ANN.predict(x)
sigma_ANN = PredVar(nbrs,ANN,y_pred_ANN,x,y)

#now calculate the approximate prediction variance using NN

#SVM
SVM = svm.SVR()
SVM.fit(X,y)
y_pred_SVM = SVM.predict(x)
sigma_SVM = PredVar(nbrs,SVM,y_pred_SVM,x,y)





#PLOT

print mesh_X.shape
plt.figure(figsize=[12,12])
plt.subplot(2, 2, 1)
reshape_val = int(math.sqrt(len(y_pred)))

cmap = cm.rainbow
levels = 50
clevels = 10
#first plot the real function

Z = f(x).reshape(reshape_val,reshape_val)
norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cset1 = plt.contourf(mesh_X, mesh_Y, Z,levels, cmap=cmap)
cset2 = plt.contour(mesh_X, mesh_Y, Z, 4,
                colors='k',
                hold='on')

plt.clabel(cset2, inline=1, fontsize=10)
plt.title('f(x) = '+benchmark_name)




plt.subplot(2, 2, 2)
#then plot the GP predictions
Z = y_pred.reshape(reshape_val,reshape_val)

cset1 = plt.contourf(mesh_X, mesh_Y, Z,levels, cmap=cmap)
cset2 = plt.contour(mesh_X, mesh_Y, Z, clevels,
                colors='k',
                hold='on')

plt.clabel(cset2, inline=1, fontsize=10)
plt.title('GP predictions')




plt.subplot(2, 2, 3)
# Then plot the RF predictions
Z = y_pred_RF.reshape(reshape_val,reshape_val)
cset1 = plt.contourf(mesh_X, mesh_Y, Z,levels, cmap=cmap)
cset2 = plt.contour(mesh_X, mesh_Y, Z, clevels,
                colors='k',
                hold='on')

plt.clabel(cset2, inline=1, fontsize=10)
plt.title('RF predictions')



plt.subplot(2, 2, 4)
# Then plot the GP sigma using heuristics
Z = y_pred_ANN.reshape(reshape_val,reshape_val)
cset1 = plt.contourf(mesh_X, mesh_Y, Z,levels, cmap=cmap)
cset2 = plt.contour(mesh_X, mesh_Y, Z, clevels,
                colors='k',
                hold='on')

plt.clabel(cset2, inline=1, fontsize=10)
plt.title('ANN predictions')
plt.tight_layout()
plt.savefig("predvar_2d_"+benchmark_name+"_predictions")

exit()


####################################################################################################################################
# New figure for random forest stuff

plt.figure(figsize=[12,12])
plt.subplot(2, 2, 1)
reshape_val = int(math.sqrt(len(y_pred)))


#first plot the real function

Z = f(x).reshape(reshape_val,reshape_val)
norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cset1 = plt.contourf(mesh_X, mesh_Y, Z,levels, cmap=cmap)
cset2 = plt.contour(mesh_X, mesh_Y, Z, 4,
                colors='k',
                hold='on')

plt.clabel(cset2, inline=1, fontsize=10)
plt.title('f(x) = '+benchmark_name)




plt.subplot(2, 2, 2)
#then plot the GP predictions
Z = y_pred_RF.reshape(reshape_val,reshape_val)

cset1 = plt.contourf(mesh_X, mesh_Y, Z,levels, cmap=cmap)
cset2 = plt.contour(mesh_X, mesh_Y, Z, clevels,
                colors='k',
                hold='on')

plt.clabel(cset2, inline=1, fontsize=10)
plt.title('RF predictions')


plt.subplot(2, 2, 3)
# Then plot the GP sigma



# Then plot the GP sigma using heuristics
Z = sigma_RF.reshape(reshape_val,reshape_val)
cset1 = plt.contourf(mesh_X, mesh_Y, Z,levels, cmap=cmap)
cset2 = plt.contour(mesh_X, mesh_Y, Z, clevels,
                colors='k',
                hold='on')

plt.clabel(cset2, inline=1, fontsize=10)
plt.title('ANN k-NN var')
plt.tight_layout()
plt.savefig("predvar_2d_"+benchmark_name+"_ANN")


