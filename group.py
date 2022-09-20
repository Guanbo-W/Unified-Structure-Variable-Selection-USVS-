# Group Selection [two groups, each group has 3 variables, true is one group is selected p=10] compare to group Lasso.
import numpy as np
from itertools import product
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
import gurobipy as gp
from gurobipy import GRB
# pip install groupyr (in terminal)
from groupyr import SGLCV
from groupyr import SGL

# Parameters specification
dim=10
n=2000
iterate=100
con_var=6
restp=dim-con_var
truebetas = np.concatenate([np.array([1,1,1,0,0,0]),np.repeat(0, restp)])
c1=3
F1=[0,1,2]
c2=3
F2=[3,4,5]
comF=list(range(con_var,dim))
MSE,JDR,MR,FAR,C,MSE_lasso,JDR_lasso,MR_lasso,FAR_lasso,C_lasso,MSE_1se,JDR_1se,MR_1se,FAR_1se,C_1se,MSE_1se_lasso,JDR_1se_lasso,MR_1se_lasso,FAR_1se_lasso,C_1se_lasso=(np.empty(iterate) for i in range(20))

# Functions
def Gen_group(n, restp):
    np.random.seed(i)
    mu1 = np.array([0, 0, 0])
    pho1 = 0.5
    r1 = np.array([
        [pho1, pho1** 2, pho1** 3],
        [pho1** 2, pho1, pho1** 2],
        [pho1** 3, pho1** 2, pho1]
    ])
    rng = np.random.default_rng()
    xx1 = rng.multivariate_normal(mu1, r1, size=n)

    mu2 = np.array([0, 0, 0])
    pho2 = 0.7
    r2 = np.array([
        [pho2, pho2 ** 2, pho2 ** 3],
        [pho2 ** 2, pho2, pho2 ** 2],
        [pho2 ** 3, pho2 ** 2, pho2]
    ])
    rng = np.random.default_rng()
    xx2 = rng.multivariate_normal(mu2, r2, size=n)

    x = np.empty((n, restp))
    for j in range(restp):
        x[:, j] = np.random.normal(0, 0.1, n)
    X = np.concatenate((xx1, xx2, x), axis=1)
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

    # add binary variables for the "OR" constraints
    or1 = regressor.addVars(2, vtype=GRB.BINARY, name="or1")
    or2 = regressor.addVars(2, vtype=GRB.BINARY, name="or2")

    # Objective Function (OF): minimize 1/2 * RSS using the fact that
    # if x* is a minimizer of f(x), it is also a minimizer of k*f(x) iff k > 0
    Quad = np.dot(X.T, X)
    lin = np.dot(response.T, X)
    obj = sum(0.5 * Quad[i,j] * beta[i] * beta[j]
              for i, j in product(range(dim + 1), repeat=2))
    obj -= sum(lin[i] * beta[i] for i in range(dim + 1))
    obj += 0.5 * np.dot(response, response)
    regressor.setObjective(obj, GRB.MINIMIZE)

    # Constraint sets
    for i in range(dim):
        # If iszero[i]=1, then beta[i] = 0
        regressor.addSOS(GRB.SOS_TYPE1, [beta[i], iszero[i]])
    #select 0 or c1 variables in F1
    regressor.addConstr((or1[0] == 1) >> (sum(iszero[j] for j in F1) == 0))
    regressor.addConstr((or1[1] == 1) >> (sum(iszero[j] for j in F1) == c1))
    regressor.addConstr(or1[0] + or1[1] == 1)
    # select 0 or c1 variables in F2
    regressor.addConstr((or2[0] == 1) >> (sum(iszero[j] for j in F2) == 0))
    regressor.addConstr((or2[1] == 1) >> (sum(iszero[j] for j in F2) == c2))
    regressor.addConstr(or2[0] + or2[1] == 1)
    # penalize the other variables
    regressor.addConstr(sum(iszero[j] for j in comF) == len(comF) - (non_zero - (or1[0]*c1 + or2[0]*c2) ) )
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

def Results(iteration_number, truebetas, coef):
    # 2) joint detection rate
    JDR = int(sum(coef[np.where(truebetas != 0)] != 0) == len(np.concatenate(np.where(truebetas != 0))))
    # 3) missing rate
    MR = len(np.concatenate(np.where((truebetas != 0) & (coef == 0)))) / len(np.concatenate(np.where(truebetas != 0)))
    # 4) false alarm rate
    FAR = len(np.concatenate(np.where((truebetas == 0) & (coef != 0)))) / len(np.concatenate(np.where(truebetas == 0)))
    # 5) consistency of respecting the selection rule
    R1=len(np.concatenate(np.where(coef[0:3]==0)))
    R2=len(np.concatenate(np.where(coef[3:6]==0)))
    if R1== c1 or R1== 0:
        C1 = 1/2
    else:
        C1 = 0
    if R2== c2 or R2== 0:
        C2 = 1/2
    else:
        C2 = 0
    C=C1+C2
    return JDR, MR, FAR, C

# Iteration
for i in range(iterate):
    # Load data and split into train (80%) and test (20%)
    np.random.seed(i)
    y, X, warm=Gen_group(n, restp)
    intercept, beta, best_mse = L0_regression(c1, F1, c2, F2, comF, X, y, warm_up=warm, seed=i)
    intercept_1se, beta_1se, best_mse_1se = L0_regression_1se(c1, F1, c2, F2, comF, X, y, warm_up=warm, seed=i)

    # unit rule min
    MSE[i]=best_mse
    JDR[i], MR[i], FAR[i], C[i]=Results(i, truebetas, beta)

    # unit rule 1se
    MSE_1se[i] = best_mse_1se
    JDR_1se[i], MR_1se[i], FAR_1se[i], C_1se[i] = Results(i, truebetas, beta_1se)


    # group Lasso min
    # weight is sqaure root p for each group (by setting scale_l2_by as None)
    groups=[np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6]), np.array([7]), np.array([8]), np.array([9])]
    lasso = SGLCV(
        groups=groups, l1_ratio=0, tol=1e-3, normalize=True, cv=5 ,scoring="neg_mean_squared_error"
    ).fit(X,y)
    mse_cv=np.mean(-lasso.scoring_path_,axis=1)

    MSE_lasso[i] = mse_cv[lasso.alphas_==lasso.alpha_]
    JDR_lasso[i], MR_lasso[i], FAR_lasso[i], C_lasso[i] = Results(i, truebetas, lasso.coef_)

    # group Lasso 1se:
    sd = np.std(-lasso.scoring_path_, axis=1)
    se = min(mse_cv) + sd[mse_cv == min(mse_cv)][0] / np.sqrt(5)
    lambda_1se = max(lasso.alphas_[mse_cv <= se])
    model=SGL(groups=groups,l1_ratio=0,alpha=lambda_1se)
    lasso_1se_coef=model.fit(X, y).coef_

    MSE_1se_lasso[i] = mse_cv[lasso.alphas_==lambda_1se]
    JDR_1se_lasso[i], MR_1se_lasso[i], FAR_1se_lasso[i], C_1se_lasso[i] = Results(i, truebetas, lasso_1se_coef)

# Print results
print(mean(MSE),mean(JDR), mean(MR), mean(FAR), mean(C))
print(mean(MSE_1se),mean(JDR_1se), mean(MR_1se), mean(FAR_1se), mean(C_1se))
print(mean(MSE_lasso),mean(JDR_lasso), mean(MR_lasso), mean(FAR_lasso), mean(C_lasso))
print(mean(MSE_1se_lasso),mean(JDR_1se_lasso), mean(MR_1se_lasso), mean(FAR_1se_lasso), mean(C_1se_lasso))