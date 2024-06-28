import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import sys

# 터미널 출력 인코딩 설정 (Windows에서는 필요할 수 있음)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 데이터 로드 (헤더가 없음을 명시)
df = pd.read_csv('post_list.csv', encoding='cp949', header=None)

# 열 이름 지정
df.columns = ['Title', 'AuthorGrade']

# 텍스트 데이터와 레이블 분리
texts = df['Title'].values
labels = df['AuthorGrade'].values

# 텍스트 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 시퀀스 패딩
max_len = 100  # 시퀀스의 최대 길이
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# 학습 데이터와 검증 데이터로 분리
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# 모델 생성
model = DecisionTreeClassifier()

# 모델 학습
model.fit(X_train, y_train)

# 검증 데이터로 평가
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
print('accuracy:', accuracy_score(y_val, y_pred))

# 모델과 토크나이저 저장
model_save_path = 'model.pkl'
tokenizer_save_path = 'tokenizer.pickle'

with open(model_save_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(tokenizer_save_path, 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

print(f"모델이 {model_save_path}에 저장되었습니다.")
print(f"토크나이저가 {tokenizer_save_path}에 저장되었습니다.")

# 저장된 모델과 토크나이저 로드
with open(model_save_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(tokenizer_save_path, 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

# 사용자 입력 받아 예측
def predict_input_text():
    input_text = input("예측할 텍스트를 입력하세요: ")
    sequence = loaded_tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = loaded_model.predict(padded_sequence)
    predicted_grade = prediction[0]
    grade_mapping = {0: '진드기', 1: '닭둘기', 2: '왁무새', 3: '침팬치', 4: '느그자'}
    grade_text = grade_mapping.get(predicted_grade, "Unknown")
    print(f'입력한 문구: {input_text}\n당신의 등급은 {grade_text}입니다.')

predict_input_text()
