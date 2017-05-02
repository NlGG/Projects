import numpy as np
import pandas as pd
from patsy import dmatrices

#　統計用ツール
import statsmodels.api as sm
import statsmodels.tsa.api as tsa

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

def rcm_model(datas, vars=['P', 'S', 'L', 'R', 'RW', 'A', 'TS', 'TT', 'WOOD', 'SOUTH', 'CMD', 'IDD', 'FAR', 'X', 'Y']):
    eq = fml_build(vars)
    y, X = dmatrices(eq, data=datas, return_type='dataframe')
    lat = X["X"]
    lon = X["Y"]
    X = X.drop(["X", "Y"], axis=1)
    names = X.columns
    for n in names:
        X[n+"_lat"] = X[n]*lat
        X[n+"_lon"] = X[n]*lon
        X[n+"_lat2"] = X[n]*lat**2
        X[n+"_lon2"] = X[n]*lon**2
        X[n+"_lat3"] = X[n]*lat**3
        X[n+"_lon3"] = X[n]*lon**3
        X[n+"_lat4"] = X[n]*lat**4
        X[n+"_lon4"] = X[n]*lon**4
        X[n+"_lat5"] = X[n]*lat**5
        X[n+"_lon5"] = X[n]*lon**5
        X[n+"_lat6"] = X[n]*lat**6
        X[n+"_lon6"] = X[n]*lon**6
        X[n+"_lat7"] = X[n]*lat**7
        X[n+"_lon7"] = X[n]*lon**7
        X[n+"_lat8"] = X[n]*lat**8
        X[n+"_lon8"] = X[n]*lon**8
        X[n+"_lat9"] = X[n]*lat**9
        X[n+"_lon9"] = X[n]*lon**9
        X[n+"_latlon"] = X[n]*lat*lon
        X[n+"_latlon12"] = X[n]*lat*lon**2
        X[n+"_latlon21"] = X[n]*lat**2*lon
        X[n+"_latlon13"] = X[n]*lat*lon**3
        X[n+"_latlon31"] = X[n]*lat**3*lon
        X[n+"_latlon22"] = X[n]*lat**2*lon**2
        X[n+"_latlon14"] = X[n]*lat*lon**4
        X[n+"_latlon41"] = X[n]*lat**4*lon
        X[n+"_latlon23"] = X[n]*lat**2*lon**3
        X[n+"_latlon32"] = X[n]*lat**3*lon**2
        X[n+"_latlon15"] = X[n]*lat*lon**5
        X[n+"_latlon51"] = X[n]*lat**5*lon**1
        X[n+"_latlon24"] = X[n]*lat**2*lon**4
        X[n+"_latlon42"] = X[n]*lat**4*lon**2
        X[n+"_latlon33"] = X[n]*lat**3*lon**3
        X[n+"_latlon16"] = X[n]*lat*lon**6
        X[n+"_latlon61"] = X[n]*lat**6*lon
        X[n+"_latlon25"] = X[n]*lat**2*lon**5
        X[n+"_latlon52"] = X[n]*lat**5*lon**2
        X[n+"_latlon34"] = X[n]*lat**3*lon**4
        X[n+"_latlon43"] = X[n]*lat**4*lon**3
        X[n+"_latlon17"] = X[n]*lat*lon**7
        X[n+"_latlon71"] = X[n]*lat**7*lon
        X[n+"_latlon26"] = X[n]*lat**2*lon**6
        X[n+"_latlon62"] = X[n]*lat**6*lon**2
        X[n+"_latlon35"] = X[n]*lat**3*lon**5
        X[n+"_latlon53"] = X[n]*lat**5*lon**3
        X[n+"_latlon44"] = X[n]*lat**4*lon**4
        X[n+"_latlon18"] = X[n]*lat*lon**8
        X[n+"_latlon81"] = X[n]*lat**8*lon**1
        X[n+"_latlon27"] = X[n]*lat**2*lon**7
        X[n+"_latlon72"] = X[n]*lat**7*lon**2
        X[n+"_latlon36"] = X[n]*lat**3*lon**6
        X[n+"_latlon63"] = X[n]*lat**6*lon**3
        X[n+"_latlon45"] = X[n]*lat**4*lon**5
        X[n+"_latlon54"] = X[n]*lat**5*lon**4
    model = sm.OLS(np.log(y), X, intercept=False)
    reg = model.fit()
    
    params = {}
    for n in names:
        p = 0
        p += reg.params[n]*np.ones(len(X))
        p += np.array(reg.params[n+"_lat"]*lat)
        p += np.array(reg.params[n+"_lon"]*lon)
        p += np.array(reg.params[n+"_latlon"]*lat*lon)
        p += np.array(reg.params[n+"_lat2"]*lat**2)
        p += np.array(reg.params[n+"_lon2"]*lon**2)
        p += np.array(reg.params[n+"_lat3"]*lat**3)
        p += np.array(reg.params[n+"_lon3"]*lon**3)
        p += np.array(reg.params[n+"_lat4"]*lat**4)
        p += np.array(reg.params[n+"_lon4"]*lon**4)
        p += np.array(reg.params[n+"_lat5"]*lat**5)
        p += np.array(reg.params[n+"_lon5"]*lon**5)
        p += np.array(reg.params[n+"_lat6"]*lat**6)
        p += np.array(reg.params[n+"_lon6"]*lon**6)
        p += np.array(reg.params[n+"_lat7"]*lat**7)
        p += np.array(reg.params[n+"_lon7"]*lon**7)
        p += np.array(reg.params[n+"_lat8"]*lat**8)
        p += np.array(reg.params[n+"_lon8"]*lon**8)
        p += np.array(reg.params[n+"_lat9"]*lat**9)
        p += np.array(reg.params[n+"_lon9"]*lon**9)
        p += np.array(reg.params[n+"_latlon12"]*lat*lon**2)
        p += np.array(reg.params[n+"_latlon21"]*lat**2*lon)
        p += np.array(reg.params[n+"_latlon13"]*lat*lon**3)
        p += np.array(reg.params[n+"_latlon31"]*lat**3*lon)
        p += np.array(reg.params[n+"_latlon22"]*lat**2*lon**2)
        p += np.array(reg.params[n+"_latlon14"]*lat*lon**4)
        p += np.array(reg.params[n+"_latlon41"]*lat**4*lon)
        p += np.array(reg.params[n+"_latlon23"]*lat**2*lon**3)
        p += np.array(reg.params[n+"_latlon32"]*lat**3*lon**2)
        p += np.array(reg.params[n+"_latlon15"]*lat*lon**5)
        p += np.array(reg.params[n+"_latlon51"]*lat**5*lon)
        p += np.array(reg.params[n+"_latlon24"]*lat**2*lon**4)
        p += np.array(reg.params[n+"_latlon42"]*lat**4*lon**2)
        p += np.array(reg.params[n+"_latlon33"]*lat**3*lon**3)
        p += np.array(reg.params[n+"_latlon16"]*lat*lon**6)
        p += np.array(reg.params[n+"_latlon61"]*lat**6*lon)
        p += np.array(reg.params[n+"_latlon25"]*lat**2*lon**5)
        p += np.array(reg.params[n+"_latlon52"]*lat**5*lon**2)
        p += np.array(reg.params[n+"_latlon34"]*lat**3*lon**4)
        p += np.array(reg.params[n+"_latlon43"]*lat**4*lon**3)
        p += np.array(reg.params[n+"_latlon17"]*lat**1*lon**7)
        p += np.array(reg.params[n+"_latlon71"]*lat**7*lon**1)
        p += np.array(reg.params[n+"_latlon26"]*lat**2*lon**6)
        p += np.array(reg.params[n+"_latlon62"]*lat**6*lon**2)
        p += np.array(reg.params[n+"_latlon35"]*lat**3*lon**5)
        p += np.array(reg.params[n+"_latlon53"]*lat**5*lon**3)
        p += np.array(reg.params[n+"_latlon44"]*lat**4*lon**4)
        p += np.array(reg.params[n+"_latlon18"]*lat**1*lon**8)
        p += np.array(reg.params[n+"_latlon81"]*lat**8*lon**1)
        p += np.array(reg.params[n+"_latlon27"]*lat**2*lon**7)
        p += np.array(reg.params[n+"_latlon72"]*lat**7*lon**2)
        p += np.array(reg.params[n+"_latlon36"]*lat**3*lon**6)
        p += np.array(reg.params[n+"_latlon63"]*lat**6*lon**3)
        p += np.array(reg.params[n+"_latlon45"]*lat**4*lon**5)
        p += np.array(reg.params[n+"_latlon54"]*lat**5*lon**4)

        params[n] = p
        
    params_df = pd.DataFrame.from_dict(params)
    return y, X, reg, params_df