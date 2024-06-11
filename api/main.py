import numpy as np
import joblib
from flask import Flask , request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from io import BytesIO



app = Flask(__name__)
CORS(app)

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load the model
random_forest_model = joblib.load("Saved_model_files/random_forest_model.pkl")
means = np.load('Saved_model_files/means.npy')
stds = np.load('Saved_model_files/stds.npy')

def standardize_user_input(user_input, means_array, stds_array):
    return (user_input - means_array) / stds_array

def predict_price(user_input):
    user_input_standardized = standardize_user_input(user_input, means, stds)

    if isinstance(user_input, list):
        result = random_forest_model.predict([user_input_standardized])
    else:
        result = random_forest_model.predict(user_input_standardized)


    return result
    print(f"The predicted price for the inputted diamond attributes is {result}")



def encode_user_input(user_input):
    color_map = {"D": 0, "E": 1, "F": 2, "G": 3, "H": 4, "I": 5, "J": 6}
    clarity_map =  {"I1": 0, "IF": 1,"SI1": 2,"SI2": 3, "VS1": 4, "VS2": 5,"VVS1": 6, "VVS2": 7}

    user_input["color"] = user_input['color'].map(color_map)
    user_input["clarity"] = user_input["clarity"].map(clarity_map)
    return user_input


def concat_df(user_input, prediced_price):
    prediced_price_df = pd.DataFrame(prediced_price, columns=['price'])
    final_price_predicted_df = pd.concat([user_input, prediced_price_df], axis=1)
    print(final_price_predicted_df)
    return final_price_predicted_df


@app.route('/predict', methods = ['POST'])
def predict():
    try:
        user_input = dict(request.json['data'])
        key_order = ['carat', 'y', 'clarity', 'color', 'z', 'x']
        ordered_user_input = dict([(k, user_input[k]) for k in key_order])
        ordered_user_input_list = list(ordered_user_input.values())
        price = predict_price(list(ordered_user_input.values()))
        return jsonify({'price' : price[0]})
    except Exception as e:
        print("exception: ",e)
        return jsonify({'error': str(e)}), 400
user_input = np.array([.83, 5.98, 3, 1, 4.43, 3.95])

@app.route("/upload", methods = ["POST"])
def handle_upload():
    try:
        print(request.files['file'])
        file = request.files['file']

        if 'file' not in request.files:
            return jsonify({'error': "No file uploaded"}), 400

        if file:
            user_input_df = pd.read_csv(file)

            encoded_user_input = encode_user_input(user_input_df.copy())

            ordered_features = ['carat', 'y', 'clarity', 'color', 'z', 'x']
            price_predicted_df = predict_price(encoded_user_input[ordered_features])

            # final_price_predicted_df
            final_predicted_df = concat_df(user_input_df, price_predicted_df)

            csv_buffer = BytesIO()
            predicted_csv_file = final_predicted_df.to_csv(index = False)
            csv_buffer.write(predicted_csv_file.encode('utf-8'))
            csv_buffer.seek(0)

            return send_file(
                csv_buffer,
                as_attachment=True,
                download_name="Predicted_prices.csv",
                mimetype='text/csv'
            ), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)



"""
color:
0 - D
1 - E
2 - F
3 - G
4 - H
5 - I
6 - J

Clarity:
0 - I1
1 - IF
2 - SI1
3 - SI2
4 - VS1
5 - VS2
6 - VVS1
7 - VVS2
"""