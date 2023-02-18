from ruamel.yaml import YAML
import pandas as pd
from encoder import OHEncoder, TargetEncoder
from preprocess import PreProcessing, load_data, train_test_split_per_time
from sklearn.metrics import classification_report
import pickle
import pandas as pd
from train_model import train_models

def main():
  
    # Load config: 
    config_path = "config.yaml"

    yaml = YAML(typ="safe")
    with open(config_path) as f:
        params = yaml.load(f)
    # load
    data = load_data(params['data_csv_path'])
    preprocessor = PreProcessing(data, params['features_to_drop'],  \
    params['features_to_fill_by_median'], params['features_to_remove_nan_values'])
    preprocessor.preprocess()
    # split data
    X_train, X_test, y_train, y_test = train_test_split_per_time(preprocessor.data)
    # Encoding
    oh_encoder = OHEncoder(params['low_cardinality_categorical_features'])
    # OneHotEnconder
    ohe = oh_encoder.get_onehot_encoder(X_train)
    # save onehot encoder
    pickle.dump(ohe, open(params['onehot_encoder_path'], 'wb'))
    encoded_oh_X_train = oh_encoder.transfrom_with_ohe(X_train, ohe)
    encoded_oh_X_test = oh_encoder.transfrom_with_ohe(X_test, ohe)
    # Target Encoder
    target_encoder = TargetEncoder()
    target_encoder_params = target_encoder.get_target_encoder_params(X_train, y_train, params['high_cardinality_categorical_features'],)
    # save target encoder
    pickle.dump(target_encoder_params, open(params['target_encoder_path'], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # The advantage of HIGHEST_PROTOCOL is that files get smaller. This makes unpickling sometimes much faster
    encoded_X_train = target_encoder.transform_with_target_encoder(encoded_oh_X_train, target_encoder_params)
    encoded_X_test = target_encoder.transform_with_target_encoder(encoded_oh_X_test, target_encoder_params)
    # Train and choose the best parameters for the model
    best_model = train_models(encoded_X_train, y_train)
    # Save model
    pickle.dump(best_model, open(params['model_path'], 'wb'))
    # Prediction
    y_pred = best_model.predict(encoded_X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()


