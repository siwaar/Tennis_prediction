import pickle
from ruamel.yaml import YAML
from encoder import FeaturesEncoder
from preprocess import PreProcessing, load_data
import pandas as pd

class ATPWinnerPredict(object):
    def __init__(self)-> None:
        # Load config: 
        config_path = "config.yaml"
        yaml = YAML(typ="safe")
        with open(config_path) as f:
            self.params = yaml.load(f)
        # data
        self.data = load_data(self.params['data_to_predict_csv_path'])
        # One Hot Encoder Encoding
        self.ohe = pickle.load(open(self.params['onehot_encoder_path'], 'rb'))
        # Target Encoder
        self.target_encoder_params = pickle.load(open(self.params['target_encoder_path'], 'rb'))
        # model
        self.model =  pickle.load(open(self.params['model_path'], 'rb'))

    def run(self) -> None:
        
        preprocessor = PreProcessing(self.data, self.params['features_to_drop'],  \
        self.params['features_to_fill_by_median'], self.params['features_to_remove_nan_values'])
        preprocessor.preprocess()
        preprocessed_data = preprocessor.data
        if 'p1_won' in self.data.columns:
            preprocessed_data = preprocessed_data.drop(columns=['p1_won'], axis=1)
        
        feature_encoder = FeaturesEncoder(self.params)
        # OneHotEnconder
        encoded_oh_X = feature_encoder.transfrom_with_ohe(preprocessed_data, self.ohe)
        # target Encoder
        encoded_X = feature_encoder.transform_with_target_encoder(encoded_oh_X, self.target_encoder_params)
        encoded_X = encoded_X.drop(columns=['tourney_date'])
        # prediction
        predictions = self.model.predict(encoded_X)
        predictions_df = pd.DataFrame(predictions, columns=['predicted_p1_won'])
        data_with_predictions = pd.concat([self.data, predictions_df],axis=1)
        # save prediction
        data_with_predictions.to_csv(self.params['predictions_csv_path'])


def main() -> None:
    ATPWinnerPredict().run()

if __name__ == '__main__':
    main()
