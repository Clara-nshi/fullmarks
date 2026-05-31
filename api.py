from flask import Flask, request, jsonify
from rf_predict_fun_py import predict_func

# 创建flask对象
app = Flask('myflask')


@app.route('/')
def index():
    return 'hello flask'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    pred = predict_func(data)
    return jsonify(pred)


@app.route('/pd', methods=['GET'])
def test():
    return 'hello world'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
