from flask import Flask, request, jsonify
from flask_cors import CORS
from rf_predict_fun_py import predict_func

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': '缺少text字段'}), 400
    text = data['text']
    if not text.strip():
        return jsonify({'error': '文本不能为空'}), 400
    try:
        result = predict_func({'text': text})
        pred_class = result.get('pred_class', '未知')
        return jsonify({'pred_class': pred_class})
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)