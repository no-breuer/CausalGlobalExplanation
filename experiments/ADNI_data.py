from imputers import CausalImputer
from permutation_estimator import PermutationEstimator
import sage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
import pandas as pd

df = pd.read_csv("C:/Users/Nils Ole Breuer/Desktop/AI/master thesis/sage-master/sage/data/ADNIMERGE.csv",
                 index_col=False)

dat = df[["AGE", "PTGENDER", "PTEDUCAT", "APOE4", "FDG", "ABETA", "PTAU", "DX"]]

dat_nona = dat.dropna()
dat_nona['DX_binary'] = np.where(dat_nona['DX'] == 'Dementia', 1, 0)
dat_ready = dat_nona.drop(['DX'], axis=1)

# encode gender
gender_df = pd.get_dummies(dat_ready['PTGENDER'])

dat_enc = pd.concat([dat_ready, gender_df], axis='columns')
dat_enc = dat_enc.drop(['PTGENDER'], axis=1)

dat_enc.loc[(dat_enc.ABETA == '>1700'), 'ABETA'] = 1700
dat_enc.loc[(dat_enc.PTAU == '>120'), 'PTAU'] = 120
dat_enc.loc[(dat_enc.PTAU == '<8'), 'PTAU'] = 8

dat_enc = dat_enc.astype('float64')
normalized_df = (dat_enc - dat_enc.min()) / (dat_enc.max() - dat_enc.min())

print(normalized_df)

dat_enc = normalized_df

column_names = list(dat_enc.columns.values)
x = dat_enc.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.columns = column_names

X = df.drop(['DX_binary'], axis=1)
y = df[['DX_binary']]

print(X.columns.values)

X = X.to_numpy()
y = y.to_numpy()

num_exp = 1
num_features = 8

SAGE_values = dict()
CausalSAGE_values = dict()

for idx in range(1, num_features + 1):
    SAGE_values[idx] = []
    CausalSAGE_values[idx] = []

for experiment_idx in range(num_exp):

    print("Round:", experiment_idx + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    # Create and fit multi-layer perceptron
    # clf = MLPClassifier(hidden_layer_sizes=(64, 128, 128, 64, 32), solver='adam', max_iter=500, random_state=1)
    # clf.fit(X_train, np.ravel(y_train))
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))

    # Create and fit Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, np.ravel(y_train))
 
    # y_pred = rf.predict(X_test)
    # print(accuracy_score(y_test, y_pred))

    imputer = sage.MarginalImputer(rf, X_test)
    causal_imputer = CausalImputer(rf, X_test, ordering=[[0, 1, 2, 6, 7], [4], [3, 5]], confounding=[True, False, True])
    estimator = PermutationEstimator(imputer, 'cross entropy')
    causal_estimator = PermutationEstimator(causal_imputer, 'cross entropy')

    sage_values = estimator(X_test, y_test)
    causal_sage = causal_estimator(X_test, y_test)

    for idx in range(1, num_features + 1):
        SAGE_values[idx].append(sage_values.values[idx - 1])
        CausalSAGE_values[idx].append(causal_sage.values[idx - 1])

df_sage = pd.DataFrame.from_dict(SAGE_values)
df_sage['method'] = 'SAGE'
df_causal = pd.DataFrame.from_dict(CausalSAGE_values)
df_causal['method'] = 'Causal SAGE'

results = pd.concat([df_sage, df_causal], ignore_index=True, axis=0)
results.to_csv('result_ADNI_RF.csv')
