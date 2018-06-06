# The k-NN uncertainty measure
Model independent heuristic estimation of prediction errors


[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

For most regression models, their overall accuracy can be estimated with help of various error measures. However, in some applications it is important to provide not only point predictions, but also to estimate the ``uncertainty'' of the prediction, e.g., in terms of confidence intervals, variances, or interquartile ranges.  There are very few statistical modeling techniques able to achieve this. For instance, the Kriging/Gaussian Process method is equipped with a theoretical mean squared error. In this paper we address this problem by introducing a heuristic method to estimate the uncertainty of the prediction, based on the error information from the k-nearest neighbours. This heuristic, called the **k-NN uncertainty measure**, is computationally much cheaper than other approaches (e.g., bootstrapping) and can be applied regardless of the underlying regression model. To validate and demonstrate the usefulness of the proposed heuristic, it is combined with various models and plugged into the well-known Efficient Global Optimization algorithm (EGO). Results demonstrate that using different models with the proposed heuristic can improve the convergence of EGO significantly.

See our paper for additional details: https://link.springer.com/chapter/10.1007/978-3-319-91479-4_40

## Equation

The equation of the **k-NN uncertainty measure** is given in latex formula below:

<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{U}_{\footnotesize&space;k\text{-NN}}&space;=&space;\underbrace{\frac{\sum\limits_{i\in&space;N(\mathbf{x})}&space;w_{i}^k\left|\hat{f}(\mathbf{x})&space;-&space;y_i\right|}{\sum\limits_{i\in&space;N(\mathbf{x})}&space;w_{i}^k}}_{\text{empirical&space;prediction&space;error}}&space;&plus;&space;\underbrace{\frac{\min\limits_{i\in&space;N(\mathbf{x})}d(\mathbf{x}_i,\mathbf{x})}{\max\limits_{\mathbf{x}_i,&space;\mathbf{x}_j\in\mathcal{X}}d(\mathbf{x}_i,&space;\mathbf{x}_j)}\widehat{\sigma}}_{\text{variability&space;of&space;the&space;observation}}.&space;\label{eq:stein}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{U}_{\footnotesize&space;k\text{-NN}}&space;=&space;\underbrace{\frac{\sum\limits_{i\in&space;N(\mathbf{x})}&space;w_{i}^k\left|\hat{f}(\mathbf{x})&space;-&space;y_i\right|}{\sum\limits_{i\in&space;N(\mathbf{x})}&space;w_{i}^k}}_{\text{empirical&space;prediction&space;error}}&space;&plus;&space;\underbrace{\frac{\min\limits_{i\in&space;N(\mathbf{x})}d(\mathbf{x}_i,\mathbf{x})}{\max\limits_{\mathbf{x}_i,&space;\mathbf{x}_j\in\mathcal{X}}d(\mathbf{x}_i,&space;\mathbf{x}_j)}\widehat{\sigma}}_{\text{variability&space;of&space;the&space;observation}}.&space;\label{eq:stein}" title="\widehat{U}_{\footnotesize k\text{-NN}} = \underbrace{\frac{\sum\limits_{i\in N(\mathbf{x})} w_{i}^k\left|\hat{f}(\mathbf{x}) - y_i\right|}{\sum\limits_{i\in N(\mathbf{x})} w_{i}^k}}_{\text{empirical prediction error}} + \underbrace{\frac{\min\limits_{i\in N(\mathbf{x})}d(\mathbf{x}_i,\mathbf{x})}{\max\limits_{\mathbf{x}_i, \mathbf{x}_j\in\mathcal{X}}d(\mathbf{x}_i, \mathbf{x}_j)}\widehat{\sigma}}_{\text{variability of the observation}}. \label{eq:stein}" /></a>

Where 
<a href="https://www.codecogs.com/eqnedit.php?latex=w_i&space;=1&space;-&space;\frac{d(\mathbf{x}_i,&space;\mathbf{x})}{\sum\limits_{i\in&space;N(\mathbf{x})}d(\mathbf{x}_i,\mathbf{x})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_i&space;=1&space;-&space;\frac{d(\mathbf{x}_i,&space;\mathbf{x})}{\sum\limits_{i\in&space;N(\mathbf{x})}d(\mathbf{x}_i,\mathbf{x})}" title="w_i =1 - \frac{d(\mathbf{x}_i, \mathbf{x})}{\sum\limits_{i\in N(\mathbf{x})}d(\mathbf{x}_i,\mathbf{x})}" /></a>
and 
<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{\sigma}&space;=&space;\sqrt{\Var\left[\{y_i\}_{i\in&space;N(\mathbf{x})}\cup\{\hat{f}(\mathbf{x})\}\right]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{\sigma}&space;=&space;\sqrt{\Var\left[\{y_i\}_{i\in&space;N(\mathbf{x})}\cup\{\hat{f}(\mathbf{x})\}\right]}" title="\widehat{\sigma} = \sqrt{\Var\left[\{y_i\}_{i\in N(\mathbf{x})}\cup\{\hat{f}(\mathbf{x})\}\right]}" /></a>



## Python examples and function

In the folders examples you can find illustrations and example code to view the effect of the uncertainty measure.  

The uncertainty measure is also given as a python function below:


```python
def knnUncertainty(k,pred,x,y):
    #The measure of how certain a give prediction is given its k neighbours
    #k is the number of neighbours taken into account
    #pred is the predicted point
    #x is the set of known points (input)
    #y is the set of known points (output)
    no = MinMaxScaler(copy=True)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(no.fit_transform(X))
    normx = no.transform(x)
    sigma = []
    distances, indices = nbrs.kneighbors(normx,k)
    
    for i in range(len(x)):
        dist = distances[i]
        ind = indices[i]
        #calculate the neirest point error
        pred_neir = pred[i]

        abs_err = np.abs(pred_neir - y[ind])
        weights = 1 - (dist / dist.sum())
        weighted_err = np.average(abs_err, weights=weights**NN) 

        nbrs_y = list(y[ind])
        nbrs_y.append(pred[i])
        nbrs_var = np.std(nbrs_y)

        min_dist = np.min(dist)
        pred_var = weighted_err + min_dist * nbrs_var
        sigma.append(pred_var)
    sigma = np.array(sigma)
    return sigma
```

## Cite our paper

If you use this uncertainty measure please cite our scientific paper:

```bibtex
@inproceedings{van2018novel,
  title={A Novel Uncertainty Quantification Method for Efficient Global Optimization},
  author={van Stein, Bas and Wang, Hao and Kowalczyk, Wojtek and B{\"a}ck, Thomas},
  booktitle={International Conference on Information Processing and Management of Uncertainty in Knowledge-Based Systems},
  pages={480--491},
  year={2018},
  organization={Springer}
}
```

