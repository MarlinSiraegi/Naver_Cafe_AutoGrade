import sys
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.tree import DecisionTreeClassifier

# 등급 매핑 사전 정의
grade_mapping = {0: '진드기', 1: '닭둘기', 2: '왁무새', 3: '침팬치', 4: '느그자'}

# 모델 및 토크나이저 로드
with open('Gradeguess_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('tokenizer.pkl', 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

# 예측 함수 정의
def predict_author_grade(text, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)
    return prediction[0]

class GradePredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Author Grade Predictor')
        
        self.layout = QVBoxLayout()

        self.text_input = QLineEdit(self)
        self.layout.addWidget(self.text_input)

        self.predict_button = QPushButton('입력', self)
        self.predict_button.clicked.connect(self.on_click)
        self.layout.addWidget(self.predict_button)

        self.result_label = QLabel('', self)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def on_click(self):
        input_text = self.text_input.text()
        self.text_input.clear()
        
        predicted_grade = predict_author_grade(input_text, loaded_model, loaded_tokenizer)
        grade_text = grade_mapping[predicted_grade]
        
        self.result_label.setText(f'입력한 문구: {input_text}\n당신의 등급은 {grade_text}입니다.')

def main():
    app = QApplication(sys.argv)
    ex = GradePredictorApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()