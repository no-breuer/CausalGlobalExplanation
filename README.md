# CausalGlobalExplanation

This is the corresponding code to the paper *Causal Global Feature Importance: Using Causal
Knowledge to Explain Model Predictions*. In this work, we added a causal framework the method *SAGE* a global explanation method for feature importance by Covert et al. (2020). Therefore, the implementation of the *SAGE* framework is used in this repository.

## Usage

To use the causal framework and compare it to classic *SAGE* one has to train an ML model on a dataset of choice and set up a causal imputer. For the causal imputer the causal ordering of the features and knowledge about confounding or interaction is necessary.

```
# Initialize data
X, Y = ....

# Create and train model
model = ....
model.fit()

# Set up causal imputer (in this step add the causal structure (ordering))
causal_imputer = CausalImputer(model, test, ordering=[], confounding=[])

# Set up estimator
causal_estimator = PermutationEstimator(causal_imputer, 'mse')

# Get feature importance values
importance_values = causal_estimator(test, Y_test)

```

## Experiments
In the experiments folder the conducted experiments of the paper can be found. To run the synthetic experiments choose in experiments.py the wished dataset and run the code.
For the ADNI experiments, first it is necessar to apply for the data on (adni.loni.usc.edu) and then run ADNI_data.py
