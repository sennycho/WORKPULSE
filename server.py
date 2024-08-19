from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

sentiment_analysis = pipeline("text-classification", "SamLowe/roberta-base-go_emotions")

negatie_emotions = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "remorse",
    "sadness",
    "nervousness"
}


def analyze_chat(chat_texts):
    negative_count = 0
    total_count = len(chat_texts)

    for text in chat_texts:
        result = sentiment_analysis(text)
        label = result[0]['label']
        score = result[0]['score']
        # 결과 출력 & 확인
        # print(f"Text: {text}")
        # print(f"Sentiment: {label}, Confidence: {score:.2f}\n")
        # ------------------
        if label in negatie_emotions:
            negative_count += 1
    
    negative_ratio = negative_count / total_count
    return negative_ratio


@app.route('/', methods=['POST'])
def analyze():
    data = request.json
    chat_texts = data.get('chats', [])

    if not chat_texts:
        return jsonify({'error': 'No chat data provided'}), 400

    negative_ratio = analyze_chat(chat_texts)

    if negative_ratio >= 0.6:  # 60프로 넘으면 많이 부정적이라고 보기 
        return jsonify({"status": "True","message": "Your Workplace Social Wellness is not good!"})
    else:
        return jsonify({"status": "False","message": "Your Workplace Social Wellness is fine."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
