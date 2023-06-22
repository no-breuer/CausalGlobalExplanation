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
import data_gen_case1, data_gen_case2, data_gen_case3, data_gen_case4, data_gen_case5, data_generator_1, data_gen_case1_1, data_gen_case1_2
import datagen_modules
import xgboost as xgb
import pandas as pd


def true_shap(dat):
    v1 = 1
    v2 = 1
    v3 = 1
    intv_vector = [v1, v2, v3]

    dim_Data = dat[0].shape[1]
    Shapley_True = dict()
    permuted = np.random.permutation([1, 2, 3])

    # Generate the set S
    Slist = []
    for idx in range(len(permuted) + 1):
        Slist.append(permuted[:idx])

    Shapley_True_experiment_idx = dict()
    for idx in range(1, dim_Data + 1):
        Shapley_True_experiment_idx[idx] = 0
        Shapley_True[idx] = 0

    for idx in range(1, dim_Data + 1):  # For all S
        # Take the variable index to be updated
        S0 = Slist[idx - 1]  # pre_{\pi}(Vi)
        S1 = Slist[idx]  # Vi and pre_{\pi}(Vi)
        updated_idx = np.setdiff1d(S1, S0)[0]  # Vi

        intv_idx = list(S0)
        intv_idx.sort()
        intv = [intv_vector[subidx - 1] for subidx in intv_idx]
        intervention = ["v" + str(subidx) for subidx in intv_idx]

        true_do_S0 = compute_do_Y(dat, intv_idx, intv)

        intv_idx = list(S1)
        intv_idx.sort()
        intv = [intv_vector[subidx - 1] for subidx in intv_idx]
        intervention = ["v" + str(subidx) for subidx in intv_idx]

        true_do_S1 = compute_do_Y(dat, intv_idx, intv)

        Shapley_True_experiment_idx[updated_idx] = true_do_S1 - true_do_S0

    for idx in range(1, dim_Data + 1):  # For all S
        Shapley_True[idx] += Shapley_True_experiment_idx[idx]

    return Shapley_True


def compute_do_Y(dat, intv_idx, intv, seednum=1):
    if intv_idx == [1]:
        v1 = intv[0]
        # Generate do(v1) data
        Data_v1 = data_gen_case1.SCM(n=100000, v1=v1, seednum=seednum)
        Y_do_v1 = Data_v1[1]
        true_do = np.mean(Y_do_v1)
    elif intv_idx == [3]:
        v3 = intv[0]
        # Generate do(v3) data
        Data_v3 = data_gen_case1.SCM(n=100000, v3=v3, seednum=seednum)
        Y_do_v3 = Data_v3[1]
        true_do = np.mean(Y_do_v3)
    elif intv_idx == [2]:
        v2 = intv[0]
        # Generate do(v2) data
        Data_v2 = data_gen_case1.SCM(n=100000, v2=v2, seednum=seednum)
        Y_do_v2 = Data_v2[1]
        true_do = np.mean(Y_do_v2)
    elif intv_idx == [1, 2]:
        v1, v2 = intv
        # Generate do(v1, v2) data
        Data_v1v2 = data_gen_case1.SCM(n=100000, v1=v1, v2=v2, seednum=seednum)
        Y_do_v1v2 = Data_v1v2[1]
        true_do = np.mean(Y_do_v1v2)
    elif intv_idx == [1, 3]:
        v1, v3 = intv
        # Generate do(v1, v3) data
        Data_v1v3 = data_gen_case1.SCM(n=100000, v1=v1, v3=v3, seednum=seednum)
        Y_do_v1v3 = Data_v1v3[1]
        true_do = np.mean(Y_do_v1v3)
    elif intv_idx == [2, 3]:
        v2, v3 = intv
        # Generate do(v2, v3) data
        Data_v2v3 = data_gen_case1.SCM(n=100000, v2=v2, v3=v3, seednum=seednum)
        Y_do_v2v3 = Data_v2v3[1]
        true_do = np.mean(Y_do_v2v3)
    elif intv_idx == [1, 2, 3]:
        v1, v2, v3 = intv
        # Generate do(v1, v2, v3) data
        Data_v1v2v3 = data_gen_case1.SCM(n=100000, v1=v1, v2=v2, v3=v3, seednum=seednum)
        Y_do_v1v2v3 = Data_v1v2v3[1]
        true_do = np.mean(Y_do_v1v2v3)
    elif intv_idx == []:
        true_do = np.mean(dat[1])

    return true_do



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
        True_values[idx] = []

    for experiment_idx in range(num_experiments):
        print("Round:", experiment_idx + 1)

        # data generation with different causal structures
        dat = data_gen_case1_1.SCM(n)
        #X, topoSort, dictName, parentDict = data_generator_1.dataGen(n, seednum=1234)
        #train, Y_train, test, Y_test = datagen_modules.dataSplit(X, seednum=1234)

        V, Y = dat

        V = np.asarray(V)
        Y = np.asarray(Y)

        #scaler = StandardScaler()
        #V_normed = scaler.fit_transform(V)

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
        #estimator = sage.PermutationEstimator(imputer, 'mse')
        causal_estimator = PermutationEstimator(causal_imputer, 'mse')

        sage_values = estimator(test, Y_test)
        causal_sage_values = causal_estimator(test, Y_test)
        #true_values = true_shap(dat)

        for idx in range(1, num_features + 1):
            SAGE_values[idx].append(sage_values.values[idx-1])
            CausalSAGE_values[idx].append(causal_sage_values.values[idx-1])
            #True_values[idx].append(true_values[idx])

    df_sage = pd.DataFrame.from_dict(SAGE_values)
    df_sage['method'] = 'SAGE'
    df_causal = pd.DataFrame.from_dict(CausalSAGE_values)
    df_causal['method'] = 'Causal SAGE'
    #df_true = pd.DataFrame.from_dict(True_values)
    #df_true['method'] = 'True'

    results = pd.concat([df_sage, df_causal], ignore_index=True, axis=0)



    results.to_csv('resultMLP_scm1.csv')

    #print(df_true)

    """
    for idx in range(1, num_features + 1):
        SAGE_values[idx] /= num_experiments
        CausalSAGE_values[idx] /= num_experiments
        True_values[idx] /= num_experiments

    print(SAGE_values)
    print(CausalSAGE_values)
    print(True_values)

    SAGE_acc = np.mean(np.abs(np.asarray(list(True_values.values())) - np.asarray(list(SAGE_values.values()))))
    CausalSAGE_acc = np.mean(
        np.abs(np.asarray(list(True_values.values())) - np.asarray(list(CausalSAGE_values.values()))))

    print(SAGE_acc)
    print(CausalSAGE_acc)
    
    """