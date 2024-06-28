import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import sys

# �͹̳� ��� ���ڵ� ���� (Windows������ �ʿ��� �� ����)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ������ �ε� (����� ������ ���)
df = pd.read_csv('post_list.csv', encoding='cp949', header=None)

# �� �̸� ����
df.columns = ['Title', 'AuthorGrade']

# �ؽ�Ʈ �����Ϳ� ���̺� �и�
texts = df['Title'].values
labels = df['AuthorGrade'].values

# �ؽ�Ʈ ��ūȭ
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# ������ �е�
max_len = 100  # �������� �ִ� ����
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# �н� �����Ϳ� ���� �����ͷ� �и�
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# �� ����
model = DecisionTreeClassifier()

# �� �н�
model.fit(X_train, y_train)

# ���� �����ͷ� ��
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
print('accuracy:', accuracy_score(y_val, y_pred))

# �𵨰� ��ũ������ ����
model_save_path = 'model.pkl'
tokenizer_save_path = 'tokenizer.pickle'

with open(model_save_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(tokenizer_save_path, 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

print(f"���� {model_save_path}�� ����Ǿ����ϴ�.")
print(f"��ũ�������� {tokenizer_save_path}�� ����Ǿ����ϴ�.")

# ����� �𵨰� ��ũ������ �ε�
with open(model_save_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(tokenizer_save_path, 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)

# ����� �Է� �޾� ����
def predict_input_text():
    input_text = input("������ �ؽ�Ʈ�� �Է��ϼ���: ")
    sequence = loaded_tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = loaded_model.predict(padded_sequence)
    predicted_grade = prediction[0]
    grade_mapping = {0: '�����', 1: '�ߵѱ�', 2: '�ι���', 3: 'ħ��ġ', 4: '������'}
    grade_text = grade_mapping.get(predicted_grade, "Unknown")
    print(f'�Է��� ����: {input_text}\n����� ����� {grade_text}�Դϴ�.')

predict_input_text()
