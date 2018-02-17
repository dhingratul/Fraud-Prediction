[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Fraud-Prediction
Fraud hosts with substantial amount of fraudulent traffic using the impression logs for selected IP addresses

# Requirements
1. python 2.7
2. sklearn 0.19 dev version or with following [fix](https://github.com/scikit-learn/scikit-learn/commit/c554aad456b6302a8dd8838769769eeecc1cf734) 
3. numpy
4. pandas

# Run
1. Install the requirements for the projects by running
2. `jupyter notebook Fraud_Prediction.ipynb` to open the Jupyter notebook

# Discussions
1. A One class SVM is used for predicting fraudulent traffic (+1)
2. Using bucketed time_stamps as features increases the test accuracy by >20%, which is a good indicator that fraudulent activities is clustered well in time. This fact can be extended to find the botnet networks.
3. Over-classification with SMOTE was tried to balance a highly imbalanced dataset, but was not useful in producing better results, hence ommitted.
