from typing import Any
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
from ruamel.yaml import YAML
import pickle 
from encoder import OHEncoder, TargetEncoder
from preprocess import PreProcessing



# Load config
config_path = "config.yaml"
yaml = YAML(typ="safe")
with open(config_path) as f:
    params = yaml.load(f)

# One Hot Encoder Encoding
ohe = pickle.load(open(params['onehot_encoder_path'], 'rb'))
# Target Encoder
target_encoder_params = pickle.load(open(params['target_encoder_path'], 'rb'))
# model
model =  pickle.load(open(params['model_path'], 'rb'))
# scaler
scaler =  pickle.load(open(params['scaler_path'], 'rb'))
# features names for prediction
features = params['features_for_prediction']


app = Flask(__name__)
CORS(app)

def predict_winner(values : list[Any])-> int:
    """ Predict if the first player will win

    Args:
        values (list[Any]): input from interface

    Returns:
        int: 1 if first player will win, 0 else
    """
    assert len(values) == len(features)
    data = pd.DataFrame([values], columns=features)
    preprocessor = PreProcessing(data, [],  \
    params['features_to_fill_by_median'], params['features_to_remove_nan_values'],
    params['features_to_log'])
    preprocessor.preprocess()
    preprocessed_data = preprocessor.data
    # Scaler
    preprocessed_data[params['features_to_scale']] = scaler.transform(preprocessed_data[params['features_to_scale']].values)
    # Encoding
    oh_encoder = OHEncoder(params['low_cardinality_categorical_features'])
    # OneHotEnconder
    encoded_oh_X = oh_encoder.transfrom_with_ohe(preprocessed_data, ohe)
    # Target Encoder
    target_encoder = TargetEncoder()
    encoded_X = target_encoder.transform_with_target_encoder(encoded_oh_X, target_encoder_params)
    encoded_X = encoded_X.drop(columns=['tourney_date'])
    # prediction
    return model.predict(encoded_X)[0]

@app.route("/", methods=['GET'])
def hello():
    return render_template('index.html')

@app.route("/api/atp_winner", methods=['POST'])
def predict():
    payload = request.json['data']
    values = [float(i) for i in payload]
    return jsonify({'prediction':predict_winner(values)})

@app.route("/atp_winner", methods=['POST','GET'])
def predict_interface():
    # sourcery skip: for-append-to-extend, identity-comprehension, list-comprehension, move-assign-in-block, simplify-generator, use-fstring-for-formatting
    if request.method != 'POST':
        return render_template('index.html')
    try :
        values = []
        result = request.form
        for feature, value in result.items():
            try:
                if feature in features_type_int: value = int(value)
                elif feature in features_type_float: value = float(value)
                elif feature in dict_features_options and value not in dict_features_options[feature]:
                    if feature in not_required_features :
                        value = 'None'
                    else :
                        return render_template('index.html', result='Please select one of the options for this feature: {}'.format(feature))
                elif feature == 'tourney_date' : value = value.replace('-', '')
                values.append(value)
            except Exception as e:
                return render_template('index.html', result='Please verify the provided information')

        prediction = predict_winner(values)
        if prediction == 1 :
            return render_template(
                    'index.html',
                    result='Our classiffier predicted that the first player will be the winner !')
        else:
            return render_template(
                'index.html',
                result='Our classiffier predicted that the second player will be the winner !')
        
    except Exception:
        return render_template('index.html')


features_type_int = ['p1_id', 'p2_id', 'match_num', 'p1_age', 'p2_age']
features_type_float = ['p1_rank', 'p2_rank', 'p1_rank_points', 'p2_rank_points', 'p1_seed', 'p2_seed', 'p1_ht', 'p2_ht']
not_required_features = ['p1_hand', 'p2_hand', 'tourney_level']
dict_features_options = {
            'best_of' : ['3','5'],
            'round' : ['R64', 'R32', 'R16', 'QF', 'R128', 'SF', 'F', 'RR', 'BR', 'Unkown'],
            'surface' :  ['Hard', 'Grass', 'Clay', 'Carpet'],
            'tourney_level':['M', 'A', 'G', 'D', 'F'],
            'p1_hand' : ['R', 'L'],
            'p2_hand': ['R', 'L'],
            'p1_ioc' :['ESP', 'CYP', 'USA', 'NED', 'SRB', 'RUS', 'CAN', 'SVK', 'GBR',
                                    'FRA', 'BEL', 'HUN', 'JPN', 'CRO', 'ARG', 'AUT', 'POL', 'ISR',
                                    'SRI', 'SLO', 'IND', 'AUS', 'RSA', 'LAT', 'LUX', 'GER', 'COL',
                                    'BOL', 'NOR', 'TPE', 'KOR', 'ITA', 'BIH', 'UKR', 'KAZ', 'TUN',
                                    'PAK', 'CZE', 'BRA', 'EST', 'URU', 'POR', 'BAR', 'SWE', 'SUI',
                                    'ROU', 'BUL', 'PER', 'TUR', 'LTU', 'DOM', 'GRE', 'ECU', 'UZB',
                                    'PHI', 'CHI', 'BLR', 'THA', 'MDA', 'FIN', 'ZIM', 'VEN', 'INA',
                                    'MON', 'CRC', 'ALG', 'CHN', 'KUW', 'DEN', 'NZL', 'EGY', 'MEX',
                                    'GEO', 'VIE', 'PAR', 'LIB', 'MAR', 'PUR', 'ESA', 'BAH', 'MAS',
                                    'GUA', 'IRL', 'IRI', 'MAD', 'QAT', 'HKG', 'UAE'],
            'p2_ioc': ['ESP', 'CYP', 'USA', 'NED', 'SRB', 'RUS', 'CAN', 'SVK', 'GBR',
                                    'FRA', 'BEL', 'HUN', 'JPN', 'CRO', 'ARG', 'AUT', 'POL', 'ISR',
                                    'SRI', 'SLO', 'IND', 'AUS', 'RSA', 'LAT', 'LUX', 'GER', 'COL',
                                    'BOL', 'NOR', 'TPE', 'KOR', 'ITA', 'BIH', 'UKR', 'KAZ', 'TUN',
                                    'PAK', 'CZE', 'BRA', 'EST', 'URU', 'POR', 'BAR', 'SWE', 'SUI',
                                    'ROU', 'BUL', 'PER', 'TUR', 'LTU', 'DOM', 'GRE', 'ECU', 'UZB',
                                    'PHI', 'CHI', 'BLR', 'THA', 'MDA', 'FIN', 'ZIM', 'VEN', 'INA',
                                    'MON', 'CRC', 'ALG', 'CHN', 'KUW', 'DEN', 'NZL', 'EGY', 'MEX',
                                    'GEO', 'VIE', 'PAR', 'LIB', 'MAR', 'PUR', 'ESA', 'BAH', 'MAS',
                                    'GUA', 'IRL', 'IRI', 'MAD', 'QAT', 'HKG', 'UAE']

        }
   

# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)