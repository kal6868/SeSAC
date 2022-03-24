import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request

from collections import Counter
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('./dataset/ratings_train.txt')
test_data = pd.read_table('./dataset/ratings_test.txt')

train_data.info()
train_data.head() # 1:긍정, 0:부정

train_data['document'].nunique(), train_data['label'].nunique()
train_data.drop_duplicates(subset=['document'], inplace=True)

train_data['label'].value_counts()
train_data.isnull().sum()

train_data = train_data.dropna(how='any')
train_data.isnull().sum()

test_data = test_data.dropna(how='any')
test_data.isnull().sum()

train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data.head()

train_data['document'] = train_data['document'].str.replace('^ +', '')
train_data['document'].replace('', np.nan, inplace=True)
train_data.isnull().sum()

train_data = train_data.dropna(how = 'any')
train_data.isnull().sum()

test_data.drop_duplicates(subset=['document'], inplace=True)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data['document'] = test_data['document'].str.replace('^ +', '')
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how = 'any')
test_data.isnull().sum()

okt = Okt()
okt.morphs('교도소 이야기구먼 솔직히 재미는 없다평점 조정', stem=True)

X_train = []
stopwords = ['은','는','이','가','을','를','의','으로','에','와','한다','하다','한']

for data in tqdm(train_data['document']):
    tokenized_data = okt.morphs(data, stem=True)
    # 불용어처리
    _data = []
    for word in tokenized_data:
        if not word in stopwords:
            _data.append(word)
    X_train.append(_data)

train_data['tokenized'] = X_train
train_data.head()

Y_train = train_data['label'].values
Y_train

X_test =[]
for data in tqdm(test_data['document']):
    tokenized_data = okt.morphs(data, stem = True)
    _data = []
    for word in tokenized_data:
        if not word in stopwords:
            _data.append(word)
    X_test.append(_data)
test_data['tokenized'] = X_test
Y_test = test_data['label'].values

test_data.head()

negative_words = np.hstack(train_data[train_data['label'] == 0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data['label'] == 1]['tokenized'].values)

negative_words_cnt = Counter(negative_words)
negative_words_cnt.most_common(20)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

tokenizer.word_index

#긍정 리뷰, 부정 리뷰의 길이

#긍정 리뷰(평균 길이 10.9)
positive_text_length = train_data[train_data['label'] ==1]['tokenized'].map(lambda x :len(x))
np.mean(positive_text_length)

#긍정 리뷰(평균 길이 10.9)
negative_text_length = train_data[train_data['label'] ==0]['tokenized'].map(lambda x :len(x))
np.mean(negative_text_length)

#리뷰에 사용된 단어가 적을 경우 제외
threshold = 3
total_cnt = len(tokenizer.word_index) # 전체 단어의 수
useless_cnt = 0 #threshold(3) 보다 적게 나온 단어의 수
total_freq = 0
useless_freq = 0
#    print(f"key:{key}, value:{value}")
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    if value < threshold:
        useless_cnt = useless_cnt + 1
        useless_freq = useless_freq + value
print(f"전체 단어 수 :{total_cnt}")
print(f"사용된 회수가 2번 이하의 단어들의 수 : {useless_cnt}")
print(f"사용된 회수가 2번 미만의 단어가 사용된 비율 : {(useless_cnt/total_cnt) * 100}")
print(f"전체 단어 중에 2번 미만의 단어가 사용된 빈도 : {(useless_freq/total_freq) * 100}")

vocab_size = total_cnt - useless_cnt + 1
print(vocab_size)

tokenzier = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(f"X_train : {len(X_train)}, X_test : {len(X_test)}, Y_train : {len(Y_train)}, Y_test : {len(Y_test)}")
#for index, sentence in enumerate(X_train):
#    if len(sentence) < 1
#        print(index)

empty_data = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
len(empty_data)

X_train = np.delete(X_train, empty_data, axis = 0)
Y_train = np.delete(Y_train, empty_data, axis = 0)

max_length_x_train = max(len(review) for review in X_train)
mean_length_x_train = sum(map(len, X_train))/len(X_train)
print(f"max_length_x_train : {max_length_x_train}, mean_length_x_train:{mean_length_x_train}")

plt.hist([len(review) for review in X_train], bins = 30)
plt.xlabel('length')
plt.ylabel('count')
plt.show()

#평균으로 하면 데이터가 너무 짧아질 수 있다.
max_length = 40
X_train = pad_sequences(X_train, maxlen = max_length)
X_test = pad_sequences(X_test, maxlen = max_length)

#Generate a model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# 1. Sequences
embedding_dim =100
V = len(tokenizer.word_index)
model_1 = Sequential()
model_1.add(Embedding(V + 1, embedding_dim))
model_1.add(LSTM(128))
model_1.add(Dense(1, activation='sigmoid'))

#optimizer = gd, sgd(local minima), momentum(minima 이전에 종료될 가능성이 있다)
# adagrad(gradient 제곱, 가중치가 0이 될 수 있다), rmsprop(adagrad의 개선 가중치가 0이 되지 않는다.)
# adam(momentum+rmsprop)
model_1.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])

# 2. Model
i = Input(shape = (max_length, ))
x = Embedding(V + 1, embedding_dim)(i)
x = LSTM(128)(x)
x = Dense(1, activation='sigmoid')(x)
model_2 = Model(i, x)
model_2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
mc = ModelCheckpoint('review_model.h5', monitor='val_acc', mode = 'max', verbose=1, save_best_only=True)

r = model_1.fit(X_train, Y_train, epochs=20, validation_split=0.2, callbacks = [es, mc])

review1 = '기나긴 서사를 잘 마무리한 작품같음. 그리고 새롭게 시작한다는 예고를 보여줘서 앞으로 후속작도 기대됨'

# 1.한글 데이터가 아닌 값을 제거
new_review = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ,가-힣]', '', review1)

# 2. 토큰화
new_review = okt.morphs(new_review, stem = True)
print(new_review)

# 3. 불용어 제거
new_review = [word for word in new_review if not word in stopwords]
print(new_review)

# 4. Embedding 처리 (text -> sequences)
new_review = tokenizer.texts_to_sequences([new_review])
print(new_review)

# 5. padding -> 길이 값을 마추기 위헤ㅐ
padding_review = pad_sequences(new_review, maxlen = max_length)

# 6. predict
score = float(model_1.predict(padding_review))

# 7. 결과 값 판단
def classfy_good_bad(socre):
    if score >= 0.5:
        print('{} 확률로 긍정 리뷰'.format(score*100))
    else:
        print('{} 확률로 부정 리뷰'.format((1 - score) * 100))

classfy_good_bad(score)

# con
def review_preidct(review):
    new_review = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ,가-힣]', '', review)
    new_review = okt.morphs(new_review, stem=True)
    new_review = [word for word in new_review if not word in stopwords]
    new_review = tokenizer.texts_to_sequences([new_review])
    padding_review = pad_sequences(new_review, maxlen=max_length)
    score = float(model_1.predict(padding_review))
    if score >= 0.5:
        print('{:.2f}% 확률로 긍정 리뷰'.format(score*100))
    else:
        print('{.2f}% 확률로 부정 리뷰'.format((1 - score) * 100))

review_preidct(review1)
#model_1.save('./best_model.h5')