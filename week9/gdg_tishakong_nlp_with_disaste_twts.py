import pandas as pd
import numpy as np
import tensorflow as tf
import re

from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Embedding, GRU, Dense, Dropout
from keras._tf_keras.keras.layers import Embedding, GRU, LSTM, Dense, Dropout
from keras._tf_keras.keras.models import Sequential

# 출력 옵션 설정 (판다스 ... 로 줄임당하는 거 볼 수 있게 하는 설정)
pd.set_option('display.max_columns', None)  # 모든 열 표시
pd.set_option('display.max_rows', None)     # 모든 행 표시
pd.set_option('display.max_colwidth', None) # 모든 데이터 전체 길이 표시

'''데이터 로드'''
train_file = "train.csv"
test_file = "test.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

'''EDA'''
train.head()
test.head()

train["length"] = train["text"].apply(lambda x : len(x))
test["length"] = test["text"].apply(lambda x : len(x))
'''
#둘다 length mean은 100정도,
print("Train Length Stat")
print(train["length"].describe())
print()
print("Test Length Stat")
print(test["length"].describe())
'''

'''데이터 전처리'''
#사진 포함 여부가 영향이 있는지 확인해보는 것도 좋을듯 (http 이렇게 시작하는 값이 있으면 IMAGEATTACHED로 치환)
# 치환 함수
def preprocess_text(text):
    text = re.sub(r'http://t\.co/\S+', 'IMAGEATTACHED', text) #이미지 치환
    text = re.sub(r'@\S+', 'MENTIONED', text) #멘션 치환
    #태그도 치환할까? 근데 태그는 내용이 중요할수도
    return text

# text 열 치환 적용 후 삭제 (text_p => 전처리 된 텍스트)
train['text_p'] = train['text'].apply(preprocess_text)
test['text_p'] = test['text'].apply(preprocess_text)
train.drop(columns=['text'], inplace=True)
test.drop(columns=['text'], inplace=True)

#print(train['text_p'].tail()) IMAGEATTACHED 제대로 변환 됐는지 확인
#대문자 소문자 통일은 어큐러시 더 떨어짐

# 토큰화 및 시퀀스 변환
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train['text_p'])

X_train = tokenizer.texts_to_sequences(train['text_p'])
X_test = tokenizer.texts_to_sequences(test['text_p'])

# 패딩
max_length = 100 #mean이었던 100으로 설정
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

# 타겟값 준비
y_train = train['target']

# GRU 모델 정의 => GRU 성능 나빠서 LSTM으로 변경 => 똑같은데요,,
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length),  # 임베딩 층
    LSTM(64, return_sequences=False),  # GRU 층
    Dropout(0.3),  # 과적합 방지를 위한 드롭아웃
    Dense(32, activation='relu'),  # 완전 연결층
    Dense(1, activation='sigmoid')  # 출력층 (이진 분류)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 구조 확인 model.summary()

# 데이터셋 분리
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 학습
history = model.fit(
    X_train_split, y_train_split,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# 테스트셋 예측
predictions = model.predict(X_test)

# 결과 확인
print(predictions[:10])  # 예측 확률 출력

# 테스트 데이터의 id 열과 예측 값 결합
output = pd.DataFrame({
    'id': test['id'],  # test 데이터셋의 id 열
    'target': (predictions > 0.5).astype(int).flatten()  # 0.5를 기준으로 이진 분류
})

# CSV 파일로 저장
output.to_csv("submission.csv", index=False)

#결과가 전부 0으로 나오는데,,,