import numpy as np
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

def makedata(data):
    CITY_NAME = data['CITY_CODE'].copy()

    CITY_NAME[CITY_NAME == 13101] = '01千代田区'
    CITY_NAME[CITY_NAME == 13102] = "02中央区"
    CITY_NAME[CITY_NAME == 13103] = "03港区"
    CITY_NAME[CITY_NAME == 13104] = "04新宿区"
    CITY_NAME[CITY_NAME == 13105] = "05文京区"
    CITY_NAME[CITY_NAME == 13106] = "06台東区"
    CITY_NAME[CITY_NAME == 13107] = "07墨田区"
    CITY_NAME[CITY_NAME == 13108] = "08江東区"
    CITY_NAME[CITY_NAME == 13109] = "09品川区"
    CITY_NAME[CITY_NAME == 13110] = "10目黒区"
    CITY_NAME[CITY_NAME == 13111] = "11大田区"
    CITY_NAME[CITY_NAME == 13112] = "12世田谷区"
    CITY_NAME[CITY_NAME == 13113] = "13渋谷区"
    CITY_NAME[CITY_NAME == 13114] = "14中野区"
    CITY_NAME[CITY_NAME == 13115] = "15杉並区"
    CITY_NAME[CITY_NAME == 13116] = "16豊島区"
    CITY_NAME[CITY_NAME == 13117] = "17北区"
    CITY_NAME[CITY_NAME == 13118] = "18荒川区"
    CITY_NAME[CITY_NAME == 13119] = "19板橋区"
    CITY_NAME[CITY_NAME == 13120] = "20練馬区"
    CITY_NAME[CITY_NAME == 13121] = "21足立区"
    CITY_NAME[CITY_NAME == 13122] = "22葛飾区"
    CITY_NAME[CITY_NAME == 13123] = "23江戸川区"

    #Make Japanese Block name
    BLOCK = data["CITY_CODE"].copy()
    BLOCK[BLOCK == 13101] = "01都心・城南"
    BLOCK[BLOCK == 13102] = "01都心・城南"
    BLOCK[BLOCK == 13103] = "01都心・城南"
    BLOCK[BLOCK == 13104] = "01都心・城南"
    BLOCK[BLOCK == 13109] = "01都心・城南"
    BLOCK[BLOCK == 13110] = "01都心・城南"
    BLOCK[BLOCK == 13111] = "01都心・城南"
    BLOCK[BLOCK == 13112] = "01都心・城南"
    BLOCK[BLOCK == 13113] = "01都心・城南"
    BLOCK[BLOCK == 13114] = "02城西・城北"
    BLOCK[BLOCK == 13115] = "02城西・城北"
    BLOCK[BLOCK == 13105] = "02城西・城北"
    BLOCK[BLOCK == 13106] = "02城西・城北"
    BLOCK[BLOCK == 13116] = "02城西・城北"
    BLOCK[BLOCK == 13117] = "02城西・城北"
    BLOCK[BLOCK == 13119] = "02城西・城北"
    BLOCK[BLOCK == 13120] = "02城西・城北"
    BLOCK[BLOCK == 13107] = "03城東"
    BLOCK[BLOCK == 13108] = "03城東"
    BLOCK[BLOCK == 13118] = "03城東"
    BLOCK[BLOCK == 13121] = "03城東"
    BLOCK[BLOCK == 13122] = "03城東"
    BLOCK[BLOCK == 13123] = "03城東"

    names = list(data.columns) + ['CITY_NAME', 'BLOCK']
    data = pd.concat((data, CITY_NAME, BLOCK), axis = 1)
    data.columns = names

    vars = ['P', 'S', 'L', 'R', 'RW', 'A', 'TS', 'TT', 'WOOD', 'SOUTH', 'CMD', 'IDD', 'FAR', 'X', 'Y']
    eq = fml_build(vars)

    y, X = dmatrices(eq, data=data, return_type='dataframe')

    CITY_NAME = pd.get_dummies(data['CITY_NAME'])
    TDQ = pd.get_dummies(data['TDQ'])

    X = pd.concat((X, CITY_NAME, TDQ), axis=1)

    datas = pd.concat((y, X), axis=1)

    return datas