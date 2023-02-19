from ruamel.yaml import YAML
from sklearn.preprocessing import RobustScaler
from encoder import OHEncoder, TargetEncoder
from preprocess import PreProcessing
from utils import load_data, train_test_split_per_time, display_classification_result
import pickle
from train_model import train_models


def main() -> None:
    """ Preprocess data, train and evaluate classification model """

    # Load config
    config_path = "config.yaml"
    yaml = YAML(typ="safe")
    with open(config_path) as f:
        params = yaml.load(f)

    # load data
    data = load_data(params['data_csv_path'])
    
    # Preprocessing
    preprocessor = PreProcessing(data, params['features_to_drop'], params['features_to_fill_by_median'],\
         params['features_to_remove_nan_values'], params['features_to_log'])
    preprocessor.preprocess(is_for_train=True)
    
    # split data and take into consideration time feature
    X_train, X_test, y_train, y_test = train_test_split_per_time(preprocessor.data)

    # Scale numerical features
    features = X_train[params['features_to_scale']]
    scaler = RobustScaler().fit(features.values)
    X_train[params['features_to_scale']] = scaler.transform(features.values)
    X_test[params['features_to_scale']] = scaler.transform(X_test[params['features_to_scale']].values)
    # save model
    pickle.dump(scaler, open(params['scaler_path'], 'wb'))

    # OneHotEnconder
    oh_encoder = OHEncoder(params['low_cardinality_categorical_features'])
    ohe = oh_encoder.get_onehot_encoder(X_train)
    # save onehot encoder for inference
    pickle.dump(ohe, open(params['onehot_encoder_path'], 'wb'))
    encoded_oh_X_train = oh_encoder.transfrom_with_ohe(X_train, ohe)
    encoded_oh_X_test = oh_encoder.transfrom_with_ohe(X_test, ohe)
    
    # Target Encoder
    target_encoder = TargetEncoder()
    target_encoder_params = target_encoder.get_target_encoder_params(X_train, y_train, params['high_cardinality_categorical_features'])
    # save target encoder for inference
    # the advantage of HIGHEST_PROTOCOL is that files get smaller. This makes unpickling sometimes much faster
    pickle.dump(target_encoder_params, open(params['target_encoder_path'], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    encoded_X_train = target_encoder.transform_with_target_encoder(encoded_oh_X_train, target_encoder_params)
    encoded_X_test = target_encoder.transform_with_target_encoder(encoded_oh_X_test, target_encoder_params)
    
    # train and choose the best parameters for the model
    best_model = train_models(encoded_X_train, y_train)
    
    # save model
    pickle.dump(best_model, open(params['model_path'], 'wb'))
    
    # prediction
    y_pred = best_model.predict(encoded_X_test)
    display_classification_result(y_test, y_pred)


if __name__ == "__main__":
    main()


