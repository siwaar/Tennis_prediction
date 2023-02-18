from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import json
import joblib
from ruamel.yaml import YAML
import pickle 
from encoder import OHEncoder, TargetEncoder
from utils import load_data
from preprocess import PreProcessing

headers = ['best_of', 'match_num', 'round', 'surface', 'tourney_date', 'tourney_level',
 'p1_age', 'p1_hand', 'p1_ht', 'p1_id', 'p1_ioc', 'p1_rank', 'p1_rank_points', 'p1_seed',
 'p2_age', 'p2_hand', 'p2_ht', 'p2_id', 'p2_ioc', 'p2_rank','p2_rank_points', 'p2_seed']

# Load config: 
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

app = Flask(__name__)
CORS(app)

def compute(values):  
    data = pd.DataFrame([values],columns=headers)
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
    return float(model.predict_proba(encoded_X))

@app.route("/", methods=['GET'])
def hello():
    return "Welcome to ATP winner predictor tool"

@app.route("/api/atp_winner", methods=['POST'])
def predict():
    payload = request.json['data']
    values = [float(i) for i in payload]
    return jsonify({'prediction':compute(values)})

@app.route("/atp_winner", methods=['POST','GET'])
def predict_interface():
    # sourcery skip: identity-comprehension, list-comprehension, move-assign-in-block, use-fstring-for-formatting
    if request.method == 'POST':
        values = []
        result = request.form
        for v in result.values():
            values.append(v)
        res = compute(values)

        return render_template('index.html', result='The first player will succeed with probability {}%'.format( 100 * round(res , 2) ) )
    return render_template('index.html')


# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)