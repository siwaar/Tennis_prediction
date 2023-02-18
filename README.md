# ATP Little Big Code Use Case : Prediction of the winning player in a tennis match
Author : [Siwar ABBES](https://www.linkedin.com/in/siwar-abbes/)


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
