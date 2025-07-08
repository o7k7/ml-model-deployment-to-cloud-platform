# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
 - Model Type: RandomForestClassifier (scikit-learn)

 - Version: 1.0

 - Development Date: 06/07/25

 - Author: Orhun Kupeli

 - Libraries: scikit-learn, pandas, numpy

## Intended Use
The primary purpose of this model is to predict whether an individual’s annual income will be above a certain threshold based on demographic and employment data.
The model has been developed for educational purposes.

## Training Data
UCI Adult Census Income (census.csv)

https://archive.ics.uci.edu/dataset/2/adult

## Evaluation Data

Evaluation data has been created by randomly selecting 20% of the original data. In order to maintain balance `stratify` 
parameter was used.

## Metrics
---- GENERAL SCORES ----
Precision: 0.7353
Recall: 0.63775
F1-Score: 0.683

Also, see `slice_output.txt` to get performance information on the data slices.

## Ethical Considerations
Using this model in real-world decisions that will affect individuals' lives carries serious risks of discrimination due to the potential biases it contains.

## Caveats and Recommendations
The boundaries of this model are based on the dataset, which means it might not reflect today's economical and social conditions.

Using SimpleInputer and K-fold cross-validation instead of a train-test split might boost the model's performance