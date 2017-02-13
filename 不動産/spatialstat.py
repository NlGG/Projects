import numpy as np
from scipy.stats import rankdata
import pandas as pd

#　統計用ツール
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from patsy import dmatrices

#　自作の空間統計用ツール
from spatialstat import *

#描画
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
import seaborn as sns
sns.set(font=['IPAmincho'])

#深層学習
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


def fml_build(varlst):
    """
    Binding OLS formula from a list of variable names
    varlst: variable names, the 1st var should be endogeneouse variable
    """
    varlst.reverse()
    fml = varlst.pop() + '~'
    while len(varlst) != 0:
        fml = fml + '+' + varlst.pop()
    return fml


def T_matrix(Data, m_T):
    d = np.array(Data)
    date = Data["published_date"]
    n = len(d[:, 1])
    T = np.array([np.zeros(n) for i in range(n)])

    rankT = rankdata(date, method='ordinal')

    for i in range(n):
        mydate = rankT[i]
        for j in range(1, m_T):
            if mydate - j > 0:
                c = np.where(rankT == mydate - j)[0][0]
                T[i, c] = 1 / m_T

    return T


def S_matrix(Data, m_S, lam):
    d = np.array(Data)
    X = Data["fX"]
    Y = Data["fY"]
    n = len(d[:, 1])
    S_dis = np.array([np.zeros(n) for i in range(n)])
    S = np.array([np.zeros(n) for i in range(n)])

    for i in range(n):
        for j in range(n):
            dist = (X[i] - X[j])**2 + (Y[i] - Y[j])**2
            if dist == 0:
                dist = 10000000
            S_dis[i, j] = dist
        rankS = rankdata(S_dis[i, :], method='ordinal')
        for j in range(m_S):
            if j <= n:
                c = np.where(rankS == j + 1)[0][0]

                S[i, c] = (lam**j) * S_dis[i, c]

    S = S / sum([lam**i for i in range(m_S)])

    return S

def LL(X, Y, Xs, Ys, error):   
    n = len(X)
    h = 0.1
    mean_of_error = np.zeros((len(Xs), len(Ys)))
    for i in range(len(Xs)):
        for j in range(len(Ys)):
            u1 = ((X-Xs[i])/h)**2 
            u2 = ((Y-Ys[j])/h)**2
            k = (0.9375*(1-((X-Xs[i])/h)**2)**2)*(0.9375*(1-((Y-Ys[j])/h)**2)**2)
            K = np.diag(k)
            indep = np.matrix(np.array([np.ones(n), X - Xs[i], Y-Ys[j]]).T)
            dep = np.matrix(np.array([error]).T)
            gls_model = sm.GLS(dep, indep, sigma=K)
            gls_results = gls_model.fit()
            mean_of_error[i, j] = gls_results.params[0]
    return mean_of_error
