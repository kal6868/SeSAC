import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### Demo 1
#Unicode 코드 수치를 출력 빈도 판정
def count_unicode(s):
    # Unicode 코드 수치를 저장할 배열 준비
    counter = np.zeros(65535)
    for i in range(len(s)):
        #각각의 문자를 Unicode 숫자로 변환
        code_value = ord(s[i]) #문자를 Unicode의 정수로 반환 ->chr(정수) -> chr(97) -> a
        if code_value > 65535:
            continue
        # 출력 빈도 확인
        counter[code_value] += 1
    # 각 요소를 문자 수로 나누어서 정규화
    counter = counter/len(s)
    return counter

# 학습 데이터
ko_str = '이것은 한국어 문자입니다.'
ja_str = 'これは日本語の文字です。'
en_str = 'This is English Sentences.'

X_train = [count_unicode(ko_str), count_unicode(ja_str), count_unicode(en_str)]
Y_train = ['ko', 'ja', 'en']

# 학습
model = GaussianNB()
model.fit(X_train, Y_train)

#평가데이터
ko_test_str = '안녕하세요.'
ja_test_str ='こんにちは。'
en_test_str = 'Hello.'

X_test = [count_unicode(ko_test_str), count_unicode(ja_test_str), count_unicode(en_test_str)]
Y_test = ['ko', 'ja', 'en']

#평가
y_pred = model.predict(X_test)
print(f"정답률 = {accuracy_score(Y_test, y_pred)}")


### Demo 2
import glob

#학습 데이터 준비
index = 0
x_train = []
y_train = []
for file in glob.glob('./dataset/train/*.txt'):
    # 언어 정보를 추출하고 레이블로 지정하기
    y_train.append(file[16:18])

    #파일 내부의 문자열을 모두 추출한 뒤 빈도 배열로 변환한 뒤 입력 데이터로 사용하기
    file_str = ''
    for line in open(file, 'r', encoding = 'UTF8'):
        file_str = file_str + line
    x_train.append(count_unicode(file_str))

# 학습
model = GaussianNB()
model.fit(x_train, y_train)

index = 0
x_test = []
y_test = []
for file in glob.glob('./dataset/test/*.txt'):
    # 언어 정보를 추출하고 레이블로 지정하기
    y_test.append(file[15:17])

    #파일 내부의 문자열을 모두 추출한 뒤 빈도 배열로 변환한 뒤 입력 데이터로 사용하기
    file_str = ''
    for line in open(file, 'r', encoding = 'UTF8'):
        file_str = file_str + line
    x_test.append(count_unicode(file_str))

#평가하기
y_pred = model.predict(x_test)
print(y_pred)
print("정답률 = ", accuracy_score(y_test, y_pred))