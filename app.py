from flask import Flask, jsonify, request
from model import data, heuristic_classifier, logistic_regression, decission_tree

app = Flask(__name__)
@app.route('/model/<string:type>', methods=['POST'])
def model_prediction():
    model_type = request.get_json()
    if model_type == 'heuristic':
        prediction = heuristic_classifier(data)
    elif model_type == 'logistic_regression':
        prediction = logistic_regression(data)
    elif model_type == 'decision_tress':
        prediction = decission_tree(data)

    else:
        return jsonify({'error': 'Invalid model type'})

    return jsonify({'prediction': int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
