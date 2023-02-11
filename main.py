from ruamel.yaml import YAML
import pandas as pd 


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
    # Load data
    data = load_data(params['data_csv_path'])
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


