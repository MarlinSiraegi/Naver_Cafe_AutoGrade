import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

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

# AdaBoost 모델 생성
model = DecisionTreeClassifier()

# 모델 학습
model.fit(X_train, y_train)



# 검증 데이터로 평가
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
print('accuracy:', accuracy_score(y_val, y_pred))

import pickle

# 모델 저장
with open('Gradeguess_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# 토크나이저 저장
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

print("모델과 토크나이저가 저장되었습니다.")

import pickle

# 모델 로드
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# 토크나이저 로드
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

print("모델과 토크나이저가 로드되었습니다.")

# 등급 매핑 사전 정의
grade_mapping = {0: '진드기', 1: '닭둘기', 2: '왁무새', 3: '침팬치', 4: '느그자'}

# 예측 함수 정의
def predict_author_grade(text, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return prediction[0]

# 예측 예시
if __name__ == "__main__":
    example_text = "이곳에 제목을 입력하세요."
    predicted_grade = predict_author_grade(example_text, loaded_model, loaded_tokenizer)
    grade_text = grade_mapping[predicted_grade]
    print(f'당신의 등급은 {grade_text}입니다.')
