# ATP Little Big Code Use Case : 
## Conduct a Data Science study on the ATP dataset to predict the winner of matches.
Author : Siwar ABBES


## Setting :
All information and parameters need to be specified in the config file : "config.yaml"

You have to specify the following parameters:

- `data_csv_path` : The path for the dataset to train and test the model, example : 'data/ATP_tweaked.csv'
- `data_to_predict_csv_path` : The path for the new datatset to predict, example : 'data/data_to_predict.csv'

## Requirements :

```
python3 -m venv tennis_env
source tennis_env/bin/activate
pip install -r requirements.txt
pip install pre-commit &&  pre-commit install
```
### Launch the pipline :

```
python main.py
```


## Explanation:
