# Causally-Aware Global Explanation

This is the corresponding code to the paper *Causally-Aware Shapley value for Global Explanation of Black-box
Predictive Models*.

In response to the ever-increasing influence, Artificial Intelligence plays in our lives and the associated need to understand it the field of eXplainable Artificial Intelligence (XAI) has become very popular in recent years. One way to explain AI models is to elucidate the predictive importance of features in a global sense. Shapley values offer the possibility, based on a well-established game theoretic approach to quantify the feature importance. Previous global explanation meth-
ods, based on Shapley values totally ignore the causal relations of input features. This leads to problematic sampling procedure assumptions that lie at the heart of the Shapley value method. This paper proposes a causally-aware global explanation framework for predictive models based on Shapley values. We introduce a novel sampling procedure for out-of-coalition features that respects the causal relations of input features. We derive a practical approach that incorporates causal knowledge into global explanation and offers the possibility to interpret the predictive feature importance considering their causal relation. We evaluate our method on synthetic data and on
real-world data. The explanations of our causally-aware global explanation method on synthetic data show that they are more intuitive and more faithful compared to previous global explanation methods.

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
For the ADNI experiments, first it is necessary to apply for the data on (adni.loni.usc.edu) and then run ADNI_data.py
