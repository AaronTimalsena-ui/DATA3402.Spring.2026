![](UTA-DataScience-Logo.png)

# Mushroom Classification Project

* **One Sentence Summary:** This repository uses the Mushroom Classification dataset from Kaggle to predict whether a mushroom is edible or poisonous using tabular machine learning models.

## Overview

This project uses the Mushroom Classification dataset from Kaggle. The goal is to predict whether a mushroom is edible or poisonous from its physical features.

The dataset is a good fit for a tabular classification project because every row is one mushroom and every column is a feature. Most of the work was cleaning the categorical data, checking the unusual `?` values, encoding the columns, and comparing a few basic classification models.

Two of the three trained models got perfect validation and test accuracy. That looked suspicious at first, so I checked for target leakage and row overlap. I did not find either one. The result seems to come from the dataset being very clean and easy to separate.

## Summary of Workdone

### Data

* Data:
  * Type: CSV file with categorical mushroom features.
  * Input: mushroom physical features such as cap shape, cap color, odor, gill size, stalk features, population, and habitat.
  * Output: mushroom class.
    * `e` = edible
    * `p` = poisonous
  * Size: 8124 rows and 23 original columns.
  * Instances:
    * Training: 4874 rows
    * Validation: 1219 rows
    * Testing: 2031 rows

The dataset does not include a separate official Kaggle test file. Because of that, I created my own training, validation, and testing splits from the available CSV file.

#### Preprocessing / Clean up

The dataset did not contain regular missing values according to `mushrooms.isnull().sum()`. However, the `stalk-root` column contained 2480 question mark values.

These question marks were treated as an unknown category. I replaced `?` with `unknown` instead of dropping the rows because removing 2480 rows would remove a large part of the dataset.

I also removed the `veil-type` column because it had only one unique value. Since every row had the same value for this column, it did not help separate edible mushrooms from poisonous mushrooms.

All feature columns were categorical, so I used one-hot encoding to convert them into numeric 0/1 columns. After encoding, the feature matrix had 116 columns.

#### Data Visualization

I used a few bar plots and tables to check how the target class relates to the features.

The class column is almost balanced. About 51.8% of the mushrooms are edible, and about 48.2% are poisonous.

Bruises, spore print color, and cap color showed useful differences between the two classes.

I also checked how many unique values each column had. That showed that `veil-type` only had one value, so I removed it before modeling.

### Problem Formulation

* Input:
  * Mushroom features after one-hot encoding.
  * Examples include cap shape, cap color, odor, gill size, stalk features, population, and habitat.

* Output:
  * Mushroom class.
  * `0` = edible
  * `1` = poisonous

* Models:
  * Majority-class baseline
  * Decision Tree
  * K-Nearest Neighbors
  * Naive Bayes (Bernoulli)

I used poisonous mushrooms as the positive class. In this problem, missing a poisonous mushroom matters more than warning too much about an edible one.

### Training

I trained the models with scikit-learn on my local machine.

I split the data into training, validation, and test sets. The training set was used to fit the models. The validation set was used to compare them. The test set was saved until the end.

I used `random_state=7` so the results would be repeatable. I also used stratified splitting so the edible and poisonous classes stayed balanced in each split.

The first split held out 25% for testing and kept 75% for training and validation. The second split took that 75% and further divided it into 80% training and 20% validation. Final split sizes: 4874 training rows, 1219 validation rows, and 2031 testing rows.

I did not use deep learning. This dataset is small, categorical, and tabular, so regular scikit-learn models made more sense.

### Performance Comparison

The main metrics were accuracy, precision, recall, and F1 score.

Recall is important in this project because missing a poisonous mushroom would be more serious than incorrectly warning about an edible mushroom.

Validation results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Majority Class Baseline | 0.517637 | 0.000000 | 0.000000 | 0.000000 |
| Decision Tree | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| K-Nearest Neighbors | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Naive Bayes | 0.937654 | 0.977612 | 0.891156 | 0.932384 |

The majority-class baseline performed much worse than the trained models. This shows that the perfect model performance was not caused by class imbalance.

Final testing results for K-Nearest Neighbors:

| Metric | Result |
|---|---:|
| Accuracy | 1.000000 |
| Precision | 1.000000 |
| Recall | 1.000000 |
| F1 Score | 1.000000 |

Testing confusion matrix:

|  | Predicted Edible | Predicted Poisonous |
|---|---:|---:|
| Actual Edible | 1052 | 0 |
| Actual Poisonous | 0 | 979 |

Because perfect accuracy can look suspicious, I checked for target leakage and row overlap. The target column was not included in the encoded features, and there was no overlap between the training, validation, and testing rows.

### Conclusions

The mushroom dataset was much easier to classify than I expected. The Decision Tree and K-Nearest Neighbors models both separated edible and poisonous mushrooms perfectly on the validation set. The K-Nearest Neighbors model also got perfect results on the test set. Naive Bayes reached about 94% accuracy.

The biggest data issue was the `?` values in `stalk-root`. I kept those rows and changed `?` into an `unknown` category instead of dropping them.

The perfect score should still be taken carefully. This does not mean the model would be perfect on every real mushroom dataset. It means this specific dataset has very strong patterns, especially in features like odor.

### Future Work

A useful next step would be to test the model on a messier mushroom dataset.

I would also try removing odor and training the models again. Odor seems very strong, so this would show how much the model depends on that one feature.

Another option would be to build a smaller model using only the most important features. That would make the result easier to explain.

## How to reproduce results

### Overview of files in repository

* `README.md`: project report and summary.
* `requirements.txt`: Python packages used in the project.
* `notebooks/Project.ipynb`: main notebook containing data loading, inspection, visualization, preprocessing, modeling, evaluation, and conclusion.
* `UTA-DataScience-Logo.png`: logo image used at the top of the README.

### Software Setup

This project uses Python with the following main packages:

* pandas
* numpy
* matplotlib
* scikit-learn
* kagglehub
* jupyter
* ipykernel

To set up the environment, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Data

The dataset is downloaded inside the notebook using KaggleHub:

```python
import kagglehub

dataset_path = kagglehub.dataset_download("uciml/mushroom-classification")
```

The original CSV file is not included in this repository. It can be downloaded from Kaggle:

https://www.kaggle.com/datasets/uciml/mushroom-classification

### Training

Open and run the notebook:

```text
notebooks/Project.ipynb
```

Run all cells from top to bottom. The notebook downloads the data, preprocesses it, trains the models, and evaluates performance.

#### Performance Evaluation

The notebook evaluates the models using:

* Accuracy
* Precision
* Recall
* F1 score
* Classification report
* Confusion matrix

The K-Nearest Neighbors model is used for the final held-out testing evaluation.

## Citations

* Kaggle. Mushroom Classification Dataset. https://www.kaggle.com/datasets/uciml/mushroom-classification
* UCI Machine Learning Repository. Mushroom Data Set.
