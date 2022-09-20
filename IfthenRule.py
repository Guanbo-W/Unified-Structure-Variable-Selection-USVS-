# If then rule [if the interaction is selected, then the two main terms must be selected, p=10] compare to the overlapping group Lasso.
import random

import numpy as np
from itertools import product
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
import gurobipy as gp
from gurobipy import GRB
import scipy
# type "pip install spams_mkl" in terminal to install the SPAMS
import spams

# Parameters specification
dim=10
n=50
iterate=100
con_var=3
restp=dim-con_var
truebetas = np.concatenate([np.array([1,1,0]),np.repeat(0, restp)])
c1=1
F1=[2]
c2=2
F2=[0,1]
comF=list(range(con_var,dim))

# for SPAMS
lambdaMax=0.5
num_threads = -1 # use all cores
loss = 'square'
tol=1e-4
# number of groups: 10
# groups: [0,2], [1,2], [2], [3], [4], [5], [6], [7], [8], [9]
# group 3 is inside of group 1 and 2, no other overlaps
# weight: sqrt(2), sqrt(2), 1, ...
eta_g = np.array([np.sqrt(1), np.sqrt(1), 1, 1, 1, 1, 1, 1, 1, 1,1])
groups = scipy.sparse.csc_matrix(np.array([
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool), dtype=bool)
groups_var = scipy.sparse.csc_matrix(np.array([
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool), dtype=bool)
graph = {'eta_g': eta_g, 'groups': groups, 'groups_var': groups_var}
regul = 'graph'




MSE,JDR,MR,FAR,C,MSE_lasso,JDR_lasso,MR_lasso,FAR_lasso,C_lasso,MSE_1se,JDR_1se,MR_1se,FAR_1se,C_1se,MSE_1se_lasso,JDR_1se_lasso,MR_1se_lasso,FAR_1se_lasso,C_1se_lasso=(np.empty(iterate) for i in range(20))

# Functions
def seq(start, end, by = None, length_out = None):
    len_provided = True if (length_out is not None) else False
    by_provided = True if (by is not None) else False
    if (not by_provided) & (not len_provided):
        raise ValueError('At least by or length_out must be provided')
    width = end - start
    eps = pow(10.0, -14)
    if by_provided:
        if (abs(by) < eps):
            raise ValueError('by must be non-zero.')
    #Switch direction in case in start and end seems to have been switched (use sign of by to decide this behaviour)
        if start > end and by > 0:
            e = start
            start = end
            end = e
        elif start < end and by < 0:
            e = end
            end = start
            start = e
        absby = abs(by)
        if absby - width < eps:
            length_out = int(width / absby)
        else:
            #by is too great, we assume by is actually length_out
            length_out = int(by)
            by = width / (by - 1)
    else:
        length_out = int(length_out)
        by = width / (length_out - 1)
    out = [float(start)]*length_out
    for i in range(1, length_out):
        out[i] += by * i
    if abs(start + by * length_out - end) < eps:
        out.append(end)
    return out

def Gen_ifthen(n, restp):
    np.random.seed(i)
    x1=np.random.normal(0, 0.1, n)
    x2 = np.random.normal(0, 0.1, n)
    x3 = x1*x2
    xx=np.array([x1,x2,x3]).T
    x = np.empty((n, restp))
    for j in range(restp):
        x[:, j] = np.random.normal(0, 0.1, n)
    X = np.concatenate((xx, x), axis=1)
    epsilon = np.random.normal(0, 0.05, n)
    y = X.dot(truebetas) + epsilon
    # seed = i
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20,
    #                                                random_state=seed)
    warm = np.empty(dim)
    for k in range(dim):
        warm_epsilon = np.random.normal(0, 2, 1)
        warm[k] = min((k - 1) * warm_epsilon, warm_epsilon)
        return y, X, warm

def miqp(c1, F1, c2, F2, comF, features, response, non_zero, warm_up, verbose=False):
    """
    Deploy and optimize the MIQP formulation of L0-Regression.
    """
    assert isinstance(non_zero, (int, np.integer))
    regressor = gp.Model()
    samples, dim = features.shape
    assert samples == response.shape[0]
    assert non_zero <= dim

    # Append a column of ones to the feature matrix to account for the y-intercept
    X = np.concatenate([features, np.ones((samples, 1))], axis=1)

    # Decision variables
    beta = regressor.addVars(dim + 1, lb=-GRB.INFINITY, name="beta") # Weights
    intercept = beta[dim] # Last decision variable captures the y-intercept
    intercept.varname = 'intercept'
    # iszero[i] = 1 if beta[i] = 0
    iszero = regressor.addVars(dim, vtype=GRB.BINARY, name="iszero")

    # Objective Function (OF): minimize 1/2 * RSS using the fact that
    # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
    Quad = np.dot(X.T, X)
    lin = np.dot(response.T, X)
    obj = sum(0.5 * Quad[i, j] * beta[i] * beta[j]
              for i, j in product(range(dim + 1), repeat=2))
    obj -= sum(lin[i] * beta[i] for i in range(dim + 1))
    obj += 0.5 * np.dot(response, response)
    regressor.setObjective(obj, GRB.MINIMIZE)

    # Constraint sets
    for i in range(dim):
        # If iszero[i]=1, then beta[i] = 0
        regressor.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
    # if |F1|=c1, then |F2|=c2
    regressor.addConstr((sum(iszero[j] for j in F1) == len(F1)-c1 ) >> (sum(iszero[j] for j in F2) == len(F2)-c2 ) )
    # penalize the other variables
    regressor.addConstr(sum(iszero[j] for j in comF) == len(comF) - (non_zero - (len(F1)-sum(iszero[j] for j in F1))  - (len(F2)-sum(iszero[j] for j in F2))  ) )
    # We may use the Lasso or prev solution with fewer features as warm start
    if warm_up is not None and len(warm_up) == dim:
        for i in range(dim):
            iszero[i].start = (abs(warm_up[i]) < 1e-6)

    if not verbose:
        regressor.params.OutputFlag = 0
   #regressor.params.timelimit = 60
    regressor.params.mipgap = 0.0001
    regressor.optimize()

    coeff = np.array([beta[i].X for i in range(dim)])
    return intercept.X, coeff

    # Define functions necessary to perform hyper-parameter tuning via cross-validation

def split_folds(features, response, train_mask):
    """
    Assign folds to either train or test partitions based on train_mask.
    """
    xtrain = features[train_mask,:]
    xtest = features[~train_mask,:]
    ytrain = response[train_mask]
    ytest = response[~train_mask]
    return xtrain, xtest, ytrain, ytest

def cross_validate(features, response, non_zero, warm_up, folds, standardize, seed):
    """
    Train an L0-Regression for each fold and report the cross-validated MSE.
    """
    if seed is not None:
        np.random.seed(seed)
    samples, dim = features.shape
    assert samples == response.shape[0]
    fold_size = int(np.ceil(samples / folds))
    # Randomly assign each sample to a fold
    shuffled = np.random.choice(samples, samples, replace=False)
    mse_cv = 0
    # Exclude folds from training, one at a time,
    #to get out-of-sample estimates of the MSE
    for fold in range(folds):
        idx = shuffled[fold * fold_size : min((fold + 1) * fold_size, samples)]
        train_mask = np.ones(samples, dtype=bool)
        train_mask[idx] = False
        xtrain, xtest, ytrain, ytest = split_folds(features, response, train_mask)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xtest = scaler.transform(xtest)
        intercept, beta = miqp(c1, F1, c2, F2, comF, xtrain, ytrain, non_zero, warm_up)
        ypred = np.dot(xtest, beta) + intercept
        mse_cv += mse(ytest, ypred) / folds
    # Report the average out-of-sample MSE
    return mse_cv

def L0_regression(c1, F1, c2, F2, comF, features, response, warm_up, folds=5, standardize=True, seed=None):
    """
    Select the best L0-Regression model by performing grid search on the budget.
    """
    dim = features.shape[1]
    best_mse = np.inf
    best = 0
    # val=np.empty(dim-c-1)
    # Grid search to find best number of features to consider
    for i in range(1, dim+1):
        val = cross_validate(features, response, i, warm_up, folds=folds,
                             standardize=standardize, seed=seed)
        if val < best_mse:
            best_mse = val
            best = i
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    intercept, beta = miqp(c1, F1, c2, F2, comF, features, response, best, warm_up)
    return intercept, beta, best_mse

def cross_validate_1se(features, response, non_zero, warm_up, folds, standardize, seed):
    """
    Train an L0-Regression for each fold and report the cross-validated MSE.
    """
    if seed is not None:
        np.random.seed(seed)
    samples, dim = features.shape
    assert samples == response.shape[0]
    fold_size = int(np.ceil(samples / folds))
    # Randomly assign each sample to a fold
    shuffled = np.random.choice(samples, samples, replace=False)
    mse_fold = np.empty(folds)
    # Exclude folds from training, one at a time,
    #to get out-of-sample estimates of the MSE
    for fold in range(folds):
        idx = shuffled[fold * fold_size : min((fold + 1) * fold_size, samples)]
        train_mask = np.ones(samples, dtype=bool)
        train_mask[idx] = False
        xtrain, xtest, ytrain, ytest = split_folds(features, response, train_mask)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xtest = scaler.transform(xtest)
        intercept, beta = miqp(c1, F1, c2, F2, comF, xtrain, ytrain, non_zero, warm_up)
        ypred = np.dot(xtest, beta) + intercept
        mse_fold[fold] = mse(ytest, ypred)
    mse_cv = np.mean(mse_fold)
    se_cv = np.std(mse_fold)/np.sqrt(folds)
    # Report the average out-of-sample MSE
    return mse_cv, se_cv

def L0_regression_1se(c1, F1, c2, F2, comF, features, response, warm_up, folds=5, standardize=True, seed=None):
    """
    Select the best L0-Regression model by performing grid search on the budget.
    """
    dim = features.shape[1]
    # best_mse = np.inf
    # val=np.empty(dim-c-1)
    # Grid search to find best number of features to consider
    mse_cv= np.empty(dim+1)
    se_cv= np.empty(dim+1)
    for i in range(1, dim+ 1):
        mse_cv[i], se_cv[i] = cross_validate_1se(features, response, i, warm_up, folds=folds,
                             standardize=standardize, seed=seed)
    se=min(mse_cv)+se_cv[mse_cv==min(mse_cv)]
    best=int(np.min(np.where(mse_cv<=se)))
    best_mse=mse_cv[best]
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    intercept, beta = miqp(c1, F1, c2, F2, comF, features, response, best, warm_up)
    return intercept, beta, best_mse

def cross_validate_OGL(features, response, Lambda, folds, standardize, seed):
    """
    Train an L0-Regression for each fold and report the cross-validated MSE.
    """
    if seed is not None:
        np.random.seed(seed)
    samples, dim = features.shape
    assert samples == response.shape[0]
    fold_size = int(np.ceil(samples / folds))
    # Randomly assign each sample to a fold
    shuffled = np.random.choice(samples, samples, replace=False)
    mse_cv = 0
    # Exclude folds from training, one at a time,
    #to get out-of-sample estimates of the MSE
    for fold in range(folds):
        idx = shuffled[fold * fold_size : min((fold + 1) * fold_size, samples)]
        train_mask = np.ones(samples, dtype=bool)
        train_mask[idx] = False
        xtrain, xtest, ytrain, ytest = split_folds(features, response, train_mask)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xtest = scaler.transform(xtest)
        Xtrain = np.asfortranarray(xtrain)
        Xtrain = np.asfortranarray(Xtrain - np.tile(np.mean(Xtrain, 0), (Xtrain.shape[0], 1)))
        Xtrain = spams.normalize(Xtrain)
        ytrain = np.asfortranarray(ytrain.reshape(int(n*(folds-1)/folds), 1))
        ytrain = np.asfortranarray(ytrain - np.tile(np.mean(ytrain, 0), (ytrain.shape[0], 1)))
        ytrain = spams.normalize(ytrain)
        beta = spams.fistaGraph(Y=ytrain, X=Xtrain, W0=np.zeros((Xtrain.shape[1], 1), order="F"), graph=graph, intercept=True, return_optim_info=False, loss=loss, regul=regul, lambda1=Lambda, numThreads=num_threads,  tol=tol)
        ypred = np.dot(xtest, beta)
        mse_cv += mse(ytest, ypred) / folds
    # Report the average out-of-sample MSE
    return mse_cv

def OGLCV(features, response, folds=5, standardize=True, seed=None, lambdaMax=lambdaMax, lambdaRatio=0.01, Nlambda=100):
    X = features
    y = response
    X = np.asfortranarray(X)
    X = np.asfortranarray(X - np.tile(np.mean(X, 0), (X.shape[0], 1)))
    X = spams.normalize(X)
    y = np.asfortranarray(y.reshape(n, 1))
    y = np.asfortranarray(y - np.tile(np.mean(y, 0), (y.shape[0], 1)))
    y = spams.normalize(y)
    W0 = np.zeros((X.shape[1], y.shape[1]), order="F")
    best_mse = np.inf
    # Grid search to find best number of features to consider
    for i in seq(lambdaRatio*lambdaMax, lambdaMax, Nlambda):
        val = cross_validate_OGL(X, y, Lambda=i, folds=folds,standardize=standardize, seed=seed)
        if val < best_mse:
            best_mse = val
            best_lambda=i
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    W = spams.fistaGraph(Y=y, X=X, W0=W0, intercept=True, graph=graph,  return_optim_info=False, loss=loss,regul=regul, lambda1=best_lambda, numThreads=num_threads, tol=tol,compute_gram=compute_gram)
    return W, best_mse

def cross_validate_OGL_1se(features, response, Lambda, folds, standardize, seed):
    """
    Train an L0-Regression for each fold and report the cross-validated MSE.
    """
    if seed is not None:
        np.random.seed(seed)
    samples, dim = features.shape
    assert samples == response.shape[0]
    fold_size = int(np.ceil(samples / folds))
    # Randomly assign each sample to a fold
    shuffled = np.random.choice(samples, samples, replace=False)
    mse_fold = np.empty(folds)
    # Exclude folds from training, one at a time,
    #to get out-of-sample estimates of the MSE
    for fold in range(folds):
        idx = shuffled[fold * fold_size : min((fold + 1) * fold_size, samples)]
        train_mask = np.ones(samples, dtype=bool)
        train_mask[idx] = False
        xtrain, xtest, ytrain, ytest = split_folds(features, response, train_mask)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xtest = scaler.transform(xtest)
        Xtrain = np.asfortranarray(xtrain)
        Xtrain = np.asfortranarray(Xtrain - np.tile(np.mean(Xtrain, 0), (Xtrain.shape[0], 1)))
        Xtrain = spams.normalize(Xtrain)
        ytrain = np.asfortranarray(ytrain.reshape(int(n*(folds-1)/folds), 1))
        ytrain = np.asfortranarray(ytrain - np.tile(np.mean(ytrain, 0), (ytrain.shape[0], 1)))
        ytrain = spams.normalize(ytrain)
        beta = spams.fistaGraph(Y=ytrain, X=Xtrain, W0=np.zeros((Xtrain.shape[1], 1), order="F"), graph=graph, return_optim_info=False, loss=loss, regul=regul, lambda1=Lambda, numThreads=num_threads,  tol=tol)
        ypred = np.dot(xtest, beta)
        mse_fold[fold] = mse(ytest, ypred)
    mse_cv = np.mean(mse_fold)
    se_cv = np.std(mse_fold)/np.sqrt(folds)
    # Report the average out-of-sample MSE
    return mse_cv, se_cv

def OGLCV_1se(features, response, folds=5, standardize=True, seed=None, lambdaMax=lambdaMax, lambdaRatio=0.01, Nlambda=100):
    X = features
    y = response
    X = np.asfortranarray(X)
    X = np.asfortranarray(X - np.tile(np.mean(X, 0), (X.shape[0], 1)))
    X = spams.normalize(X)
    y = np.asfortranarray(y.reshape(n, 1))
    y = np.asfortranarray(y - np.tile(np.mean(y, 0), (y.shape[0], 1)))
    y = spams.normalize(y)
    W0 = np.zeros((X.shape[1], y.shape[1]), order="F")
    # Grid search to find best number of features to consider
    mse_cv = np.empty(0)
    se_cv = np.empty(0)
    lambda_seq=seq(lambdaRatio*lambdaMax, lambdaMax, Nlambda)
    for lambda_temp in lambda_seq:
        mse_cv_temp, se_cv_temp = cross_validate_OGL_1se(X, y, Lambda=lambda_temp, folds=folds,standardize=standardize, seed=seed)
        mse_cv=np.append(mse_cv,mse_cv_temp)
        se_cv=np.append(se_cv,se_cv_temp)
    index=np.where((mse_cv <= min(mse_cv) + se_cv[mse_cv == min(mse_cv)])==True)[0]
    lambda_1se = max(np.array(lambda_seq)[index.astype(int)])
    best_mse=mse_cv[lambda_seq==lambda_1se]
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    W = spams.fistaGraph(Y=y, X=X, W0=W0, graph=graph, return_optim_info=False, loss=loss,regul=regul, lambda1=lambda_1se, numThreads=num_threads, tol=tol,compute_gram=compute_gram)
    return W, best_mse

def Results(iteration_number, truebetas, coef):
    # 2) joint detection rate
    JDR = int(sum(coef[np.where(truebetas != 0)] != 0) == len(np.concatenate(np.where(truebetas != 0))))
    # 3) missing rate
    MR = len(np.concatenate(np.where((truebetas != 0) & (coef == 0)))) / len(np.concatenate(np.where(truebetas != 0)))
    # 4) false alarm rate
    FAR = len(np.concatenate(np.where((truebetas == 0) & (coef != 0)))) / len(np.concatenate(np.where(truebetas == 0)))
    # 5) consistency of respecting the selection rule
    R1=len(np.concatenate(np.where(coef[2]!=0)))
    R2=len(np.concatenate(np.where(coef[0:2]!=0)))
    if R1== c1 and R2!= c2:
        C=0
    else:
        C=1
    return JDR, MR, FAR, C

# Iteration
for i in range(iterate):
    # Load data and split into train (80%) and test (20%)
    np.random.seed(i)
    y, X, warm=Gen_ifthen(n, restp)
    intercept, beta, best_mse = L0_regression(c1, F1, c2, F2, comF, X, y, warm_up=warm, seed=i)
    intercept_1se, beta_1se, best_mse_1se = L0_regression_1se(c1, F1, c2, F2, comF, X, y, warm_up=warm, seed=i)

    # SVS min
    MSE[i]=best_mse
    JDR[i], MR[i], FAR[i], C[i]=Results(i, truebetas, beta)

    # SVS 1se
    MSE_1se[i] = best_mse_1se
    JDR_1se[i], MR_1se[i], FAR_1se[i], C_1se[i] = Results(i, truebetas, beta_1se)


    # spams min
    intercept=np.ones(n).reshape(n,1)
    X=np.concatenate((X,intercept),axis=1)
    OGL_coef, OGL_best_mse=OGLCV(X, y, folds=5, standardize=True, seed=None, lambdaMax=lambdaMax, lambdaRatio=0.01, Nlambda=100)
    MSE_lasso[i] = OGL_best_mse
    JDR_lasso[i], MR_lasso[i], FAR_lasso[i], C_lasso[i] = Results(i, truebetas, OGL_coef.flatten()[:-1])

    # spams 1se
    OGL_1se_coef, OGL_1se_best_mse = OGLCV_1se(X, y, folds=5, standardize=True, seed=None, lambdaMax=lambdaMax, lambdaRatio=0.01,Nlambda=100)
    MSE_1se_lasso[i] = OGL_1se_best_mse
    JDR_1se_lasso[i], MR_1se_lasso[i], FAR_1se_lasso[i], C_1se_lasso[i] = Results(i, truebetas, OGL_1se_coef.flatten()[:-1])

# Print results
print(mean(MSE),mean(JDR), mean(MR), mean(FAR), mean(C))
print(mean(MSE_1se),mean(JDR_1se), mean(MR_1se), mean(FAR_1se), mean(C_1se))
print(mean(MSE_lasso),mean(JDR_lasso), mean(MR_lasso), mean(FAR_lasso), mean(C_lasso))
print(mean(MSE_1se_lasso),mean(JDR_1se_lasso), mean(MR_1se_lasso), mean(FAR_1se_lasso), mean(C_1se_lasso))