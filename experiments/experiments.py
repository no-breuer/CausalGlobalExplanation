from imputers import CausalImputer, MarginalImputer
from permutation_estimator import PermutationEstimator
import sage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import data_gen_case1, data_gen_case2, data_gen_case3, data_gen_case4, data_gen_case5, data_generator_1, \
    data_gen_case1_1, data_gen_case1_2
import datagen_modules
import xgboost as xgb
import pandas as pd


if __name__ == '__main__':
    n = 1000
    num_experiments = 10
    num_features = 3

    SAGE_values = dict()
    CausalSAGE_values = dict()
    True_values = dict()

    for idx in range(1, num_features + 1):
        SAGE_values[idx] = []
        CausalSAGE_values[idx] = []

    for experiment_idx in range(num_experiments):
        print("Round:", experiment_idx + 1)

        # data generation with different causal structures
        dat = data_gen_case1_1.SCM(n)

        V, Y = dat

        V = np.asarray(V)
        Y = np.asarray(Y)

        V_normed = (V - V.min(0)) / V.ptp(0)
        Y_normed = (Y - Y.min(0)) / Y.ptp(0)

        train = V_normed[:750]
        test = V_normed[-250:]

        Y_train = Y_normed[:750]
        Y_test = Y_normed[-250:]

        model = LinearRegression()
        model.fit(train, Y_train)

        imputer = MarginalImputer(model, test)
        causal_imputer = CausalImputer(model, test, ordering=[[0, 2], [1]], confounding=[True, False])
        estimator = PermutationEstimator(imputer, 'mse')
        causal_estimator = PermutationEstimator(causal_imputer, 'mse')

        sage_values = estimator(test, Y_test)
        causal_sage_values = causal_estimator(test, Y_test)

        for idx in range(1, num_features + 1):
            SAGE_values[idx].append(sage_values.values[idx - 1])
            CausalSAGE_values[idx].append(causal_sage_values.values[idx - 1])

    df_sage = pd.DataFrame.from_dict(SAGE_values)
    df_sage['method'] = 'SAGE'
    df_causal = pd.DataFrame.from_dict(CausalSAGE_values)
    df_causal['method'] = 'Causal SAGE'

    results = pd.concat([df_sage, df_causal], ignore_index=True, axis=0)

    results.to_csv('resultMLP_scm1.csv')
