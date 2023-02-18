# LittleBigCode Use Case : Prediction of the winner of a tennis match.
Author : [Siwar ABBES](https://www.linkedin.com/in/siwar-abbes/)

**The aim of this work** is to explore data from ATP tennis competitions in order to predict the results of tennis matches.

## Datasets 
The data set contains the details about ATP matches played between 04/01/2015 and 25/02/2019.

You will have more information about the dataset in the notebook.
In the config file : `config.yaml`, you need to specify :

`data_csv_path` : The path for the dataset to train and test the model, example : 'data/ATP_tweaked.csv'

## Requirements 
On Ubuntu :

```
python3 -m venv tennis_env
source tennis_env/bin/activate
pip install -r requirements.txt
pip install pre-commit &&  pre-commit install
```

On Windows :
```
python3 -m venv tennis_env
.\tennis_env\Scripts\activate 
pip install -r requirements.txt
pip install pre-commit 
pre-commit install
```
## Generate the prediction model
In order to regenerate the prediction model, you need to run :
```
python main.py
```

## Predict results for a new dataset
In the config file : `config.yaml`, you need to specify :

`data_to_predict_csv_path` : The path for the new datatset to predict, example : 'data/data_to_predict.csv' and run :

```
python code_inference.py
```

You will find predictions of our model in : `predictions/predictions.csv`

## Build Docker Image
First of all, we need to have docker installed on our machine:[link to install Docker](https://docs.docker.com/engine/install/)
To build docker image, we need to run :
```
docker build -t tennis-docker .
```

To launch docker container, we need to run :
```
docker run -d -p 8080:8080 tennis-docker
```

## Models used
We first used different methods to get the encoding and add new features and then we used LightGBM model to predict the winner of a tennis match.
You will find more documentation of the choice of different models in : `documentation`
You will find trained models in the folder : `models`

## Project description
This ML project is a supervised binary classification where our target feature called `p1_won` can have 2 values :
 - `p1_won = 1` : The first player won the match.
 - `p1_won = 0` : The second player won the match.

 To deal with our data and make them ready for the classifier model, we :
 - Drop redundant features.
 - Drop features with more than 80% missing values.
 - Drop features which cannot be known in advance of the match.
 - Deal with missing values.
 - Encode categorical features with low cardinality with OneHotEncoder.
 - ENcode categorical features with high cardinality with TargetEncoder.

To **fine tune the hyperparameters** of our classifier model, we used **Grid Search** with **Time Series Cross Validation**.

## Evaluation 
Since our dataset is **balanced**, the **metric of evaluation** we used is **F1 score** which represents both precision and recall in one metric. 
We split our dataset in 80% Train and 20% Test sets.
Results found on the test set: 
 - `F1 score = 63%` 
 - `Precision = 62.5%`
 - `Recall = 63%`


## Further Work
- Make more data preprocessing and feature engineering such as :
    - Impute missing values with the intersection of information for the same player in other matches.
    - Extract the number of matches won previously by a player.
- Apply more accurate Encoding for categorical features such as the feature 'Tourney_name' or 'Tourney_id' instead of droping this information.
- Normalize the numerical features.
- Deal with outliers.
- More consider the time dependency.
- More investigate in the fetaures description in order to extract more feature and also try a better way for feature selection.
- Make predictions during the match by adding information about the current situation.
   

