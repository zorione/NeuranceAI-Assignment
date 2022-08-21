# NeuranceAI Assignment

## Problem Statement
A new pharmaceutical startup is recently acquired by one of the world's largest MNCs. For the
acquisition process, the startup is required to tabulate all drugs that they have sold and account for each drug's effectiveness. A dedicated team has been assigned the task to analyze all the data. This data has been collected over the years and it contains data points such as the drug's name, reviews by customers, popularity and use cases of the drug, and so on. Members of this team are by the noise present in the data.

## Goal
The task is to make a sophisticated NLP-based Machine Learning model that has the mentioned features as the input. Also, to use the input to predict the base score of a certain drug in a provided case.

## Dataset Description
The dataset has the following columns:
| Variable Name | Description |
| ------------- | ----------- |
| patient_id    | ID of patients |
| name_of_drug             | Name of the drug prescribed |
| use_case_for_drug        | Purpose of the drug |
| review_by_patient | Review by patient |
| drug_approved_by_UIC | Date of approval of the drug by UIC |
| number_of_times_prescribed | Number of times the drug is prescribed |
| effectiveness_rating | Effectiveness of the drug |
| base_score | Generated Score |

## Files Overview
- `dataset/train.csv`: Contains the original training dataset
- `dataset/test.csv`: Contains the original test dataset.
- `dataset/new_df_train.csv`: Contains the lemmatized reviews (`review_by_patient`) from the `train.csv`. These lemmatized reviews have been used for the training purpose.
- `dataset/new_df_test.csv`: Contains the lemmatized reviews (`review_by_patient`) from the `test.csv` respectively. These lemmatized reviews have been used during the predictions.
- `Results.pdf`: Summarizes the performance of the different models (*with different hyper-parameters*) considered.
- `xgboost.ipynb` : Contains the code for training the XGB Regressor on the dataset with default parameters. 
-  `sample_submission.csv` : Contains the submission file based on the xgboost model.
- `hybrid_model.ipynb` : Contains the code for the hybrid model
- ` Other Models/LR_1000.ipynb` : Contains the code for the Linear Regression Model with 10000 max_features of TFIDF
- `Other Models/LR_5000.ipynb` : Contains the code for the Linear Regression Model with 5000 max_features of TFIDF
- `Other Models/TF_Multiple_5000.ipynb` : Contains the code for the neural network model with 5000 max_features of TFIDF
- `Other Models/XGB_20000.ipynb` : Contains the code for the XGB Regressor with 20000 max_features of TFIDF

 
## Exploration
- For the train.csv file:
  - **No null values** in the entire dataset
  - **Number of unique drug names are 2220**
  - **Number of unique use cases are 636**
- For the test.csv file:
  - **No null values** in the entire dataset
  - **Number of unique drug names are 1478**
  - **Number of unique use cases are 461**
  
## Approach
- Out of the 7 features in the dataset, `patient_id` and `drug_approved_by_UIC` are neglected straight-forward, primarily because of the fact that they aren't much useful in determining the `base_score`.
- Similarly, `name_of_drug` and `use_case_for_drug` are also dropped for training purposes.
- For the `review_by_patient` column, first the tag (*Adjective, Noun, Adverb, Verb*) is determined for every word. This is performed using NLTK's [pos_tag](https://www.nltk.org/api/nltk.tag.html) and [wordnet](https://www.nltk.org/howto/wordnet.html).
- This is followed by the lemmatization of the reviews (*using the tags determined in the previous step*) with the help of the [WordNetLemmatizer](https://www.nltk.org/api/nltk.stem.wordnet.html).
- The lemmatized dataset is saved (*as a checkpoint*) to reduce the training time of the model as well as to reduce the memory requirements.
- Followed by the lemmatization, the reviews are vectorized (*converted into numerical features*) either using a [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) or a [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).
- Followed by the vectorization, the dataset is curated using the **vectorized reviews** and the **two numerical features in the dataset** (`number_of_times_prescribed` and `effectiveness_rating`).
- This dataset is further divided into **training** (90%) and **validation** (10%) datasets for modelling purposes. 
- For modelling purposes, a couple of approaches are tried:
  - XGBoost Regressor: Based on [XGBoost](https://xgboost.readthedocs.io/en/latest/)
  - Linear Regression: Based on [Scikit-Learn](https://scikit-learn.org/stable/)
  - Linear Regression: Based on [Tensorflow](https://www.tensorflow.org/)
  - Standard Neural Network: Based on [Tensorflow](https://www.tensorflow.org/)
  - Hybrid Neural Network (Linear Regression (for the two numerical features) + LSTM-based model(for the vectorized reviews)): Based on [Tensorflow](https://www.tensorflow.org/)
- For performing the evaluation, [RMSE (Root Mean Squared Error)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) has been used.

## Optimal Approach & Results
- The optimal approach consists of using TF-IDF Vectorizer for vectorizing the reviews, followed by using XGBoost Regressor for modelling purposes.
- This approach gives a RMSE score of `0.13` on the training dataset and a RMSE score of `0.17` on the validation dataset.

## Additional Note
- All the models have been trained on Kaggle with 16 GB of CPU support.
