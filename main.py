from ruamel.yaml import YAML
import pandas as pd 
from preprocess import PreProcessing


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data
    """
    # load data :
    print(f''' {'-'*20} Loading data  {'-'*20}''')
    data = pd.read_csv(data_path, sep = ';')
    print('Data shape : ', data.shape)
    return data


def preprocess_data(params: dict[str, str]) -> None:
    """
        Load and process data.
        It takes as parameter a dictionary containing:
        data path and selected columns to deal with the task.

    """
    # Load data
    data = load_data(params['data_csv_path'])
    # Preprocess data
    preprocessing = PreProcessing(data, params['features_to_drop'],  \
        params['features_to_fill_by_median'], params['features_to_fill_by_new_category'])
    preprocessing.preprocess()
    return 


def main():
    # Load config: 
    config_path = "config.yaml"

    yaml = YAML(typ="safe")
    with open(config_path) as f:
        params = yaml.load(f)
    
    # Load and preprocess data :
    preprocess_data(params)

if __name__ == "__main__":
    main()


