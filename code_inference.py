import pickle
from ruamel.yaml import YAML
from encoder import FeaturesEncoder
from preprocess import PreProcessing, load_data
import pandas as pd

class ATPWinnerPredict(object):
    def __init__(self)-> None:
        pass


    def run(self) -> None:
        # Load config: 
        config_path = "config.yaml"

        yaml = YAML(typ="safe")
        with open(config_path) as f:
            params = yaml.load(f)

        data = load_data(params['data_to_predict_csv_path'])
        
        preprocessor = PreProcessing(data, params['features_to_drop'],  \
        params['features_to_fill_by_median'], params['features_to_remove_nan_values'])
        preprocessor.preprocess()
        preprocessed_data = preprocessor.data
        if 'p1_won' in data.columns:
            preprocessed_data = preprocessed_data.drop(columns=['p1_won'], axis=1)
        # Encoding
        ohe = pickle.load(open(params['onehot_encoder_path'], 'rb'))
        feature_encoder = FeaturesEncoder(params)
        # OneHotEnconder
        encoded_oh_X = feature_encoder.transfrom_with_ohe(preprocessed_data, ohe)
        # Target Encoder
        target_encoder_params = pickle.load(open(params['target_encoder_path'], 'rb') )
        encoded_X = feature_encoder.transform_with_target_encoder(encoded_oh_X, target_encoder_params)
        encoded_X = encoded_X.drop(columns=['tourney_date'])
        model =  pickle.load(open(params['model_path'], 'rb'))
        predictions = model.predict(encoded_X)
        predictions_df = pd.DataFrame(predictions, columns=['predicted_p1_won'])
        data_with_predictions = pd.concat([data, predictions_df],axis=1)
        data_with_predictions.to_csv(params['predictions_csv_path'])


def main() -> None:
    ATPWinnerPredict().run()

if __name__ == '__main__':
    main()
