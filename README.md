# Rolling Lookahead Decision Trees

This repository contains an implementation of a decision tree classifier based on the rolling subtree lookahead algorithm proposed in the paper ["Rolling Lookahead Learning for Optimal Classification Trees"](https://arxiv.org/abs/2304.10830).

## Abstract

Classification trees continue to be widely adopted in machine learning applications due to their inherently interpretable nature and scalability. We propose a rolling subtree lookahead algorithm that combines the relative scalability of the myopic approaches with the foresight of the optimal approaches in constructing trees. The limited foresight embedded in our algorithm mitigates the learning pathology observed in optimal approaches. At the heart of our algorithm lies a novel two-depth optimal binary classification tree formulation flexible to handle any loss function. We show that the feasible region of this formulation is an integral polyhedron, yielding the LP relaxation solution optimal. Through extensive computational analyses, we demonstrate that our approach outperforms optimal and myopic approaches in 808 out of 1330 problem instances, improving the out-of-sample accuracy by up to 23.6% and 14.4%, respectively.


## Important Notice

- **Binary Data**: This classifier is designed specifically for binary data but it can work for both binary & multi-class classification tasks.
- **Gurobi Solver**: You need to have the Gurobi solver installed on your computer to run this code. Make sure it's properly configured and accessible in your environment.
- **Code Quality**: Please note that the code in this repository may not be optimized or well-organized. It is provided for demonstration purposes and may be updated in the future to improve readability and efficiency.


## How to Run

To run the decision tree classifier, follow these steps:

1. Install the required dependencies by running:
   `pip install -r requirements.txt`

2. Import the `run` function from the provided module and call it with the appropriate arguments. Here's the function signature:
```python


import pandas as pd
import rollo_oct

# Load your training and test datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# note that train and test data should be in this format:
# y column as label
# ordinal indexes for features
train_data = {
    'y': np.random.choice([0, 1], size=100),  # Binary label column
    '1': np.random.rand(100),  # Feature column 1
    '2': np.random.rand(100),  # Feature column 2
    '3': np.random.rand(100)   # Feature column 3
}


# Run the classifier
result = rollo_oct.run(
        train=train_data,
        test=test_data,
        depth=2,
        criterion="gini"
)
```
Parameters for the run function is follows:

- `train`: A pandas DataFrame containing the training dataset.
- `test`: A pandas DataFrame containing the test dataset.
- `depth`: Maximum depth of the decision tree (default is 2).
- `criterion`: Splitting criterion for the decision tree can be "misclassification" or "gini"(default is "gini").
- `time_limit`: Time limit for training in seconds (default is 1800).
- `big_m`: Value of big M used in the optimization model (default is 99).

## Citation
```
@misc{organ2023rolling,
      title={Rolling Lookahead Learning for Optimal Classification Trees}, 
      author={Zeynel Batuhan Organ and Enis Kayış and Taghi Khaniyev},
      year={2023},
      eprint={2304.10830},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
