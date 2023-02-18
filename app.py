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
# features names for prediction
features = params['features_for_prediction']

app = Flask(__name__)
CORS(app)

def predict_winner(values : list[Any])-> float:
    #values = [20150816, 3, 2, 'R64', 'Hard', 'M', 30.33, 'R', 188.0, 104542, 'FRA', 19.0, 1645.0, 'None', 31.75, 'L', 188.0, 104269, 'ESP', 43.0, 995.0, 'None']
    assert len(values) == len(features)
    data = pd.DataFrame([values], columns=features)
    preprocessor = PreProcessing(data, [],  \
    params['features_to_fill_by_median'], params['features_to_remove_nan_values'])
    preprocessor.preprocess()
    preprocessed_data = preprocessor.data
    if 'p1_won' in data.columns:
        preprocessed_data = preprocessed_data.drop(columns=['p1_won'], axis=1)
    # Encoding
    oh_encoder = OHEncoder(params['low_cardinality_categorical_features'])
    # OneHotEnconder
    encoded_oh_X = oh_encoder.transfrom_with_ohe(preprocessed_data, ohe)
    # Target Encoder
    target_encoder = TargetEncoder()
    encoded_X = target_encoder.transform_with_target_encoder(encoded_oh_X, target_encoder_params)
    encoded_X = encoded_X.drop(columns=['tourney_date'])
    # prediction
    return model.predict_proba(encoded_X)[0][1]

@app.route("/", methods=['GET'])
def hello():
    return "Welcome to ATP winner predictor tool"

@app.route("/api/atp_winner", methods=['POST'])
def predict():
    payload = request.json['data']
    values = [float(i) for i in payload]
    return jsonify({'prediction':predict_winner(values)})

@app.route("/atp_winner", methods=['POST','GET'])
def predict_interface():
    # sourcery skip: for-append-to-extend, identity-comprehension, list-comprehension, move-assign-in-block, simplify-generator, use-fstring-for-formatting
    if request.method == 'POST':
        values = []
        result = request.form
        for v in result.values():
            values.append(v)
        proba_p1_won = predict_winner(values)
        if proba_p1_won > 0.5 :
            return render_template('index.html', result='The first player will succeed with probability {}%'.format( 100 * round(proba_p1_won , 2)))
        else: 
            return render_template('index.html', result='The second player will succeed with probability {}%'.format( 100 * round(1 - proba_p1_won , 2)))
    return render_template('index.html')


# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)