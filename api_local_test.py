import requests

url = "http://127.0.0.1:8001/predict"

if __name__ == '__main__':
    data = {'text': '666'}
    print(data)
    res = requests.post(url, json=data)
    print(res.json())
