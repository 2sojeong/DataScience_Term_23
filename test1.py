import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# train.csv 파일 읽기
train_df = pd.read_csv('train.csv')
test_df =  pd.read_csv('test.csv')


# feature
features = ["Book-Author", "Publisher", "User-ID", "Age", "Location"]
target = ["Book-Rating"]


우리가 정한 피쳐 = 저자, 출판사, 나이, 장소
타겟 = 점수

분류
유저 아이디 = 장소 점수 나이,
책 아이디 = 저자, 출판사, 책 제목, 발간년도

책 제목은 책아이디로 쓴다.
나이 0 244 


나이는 10세부터 100세로 설정할 것
장소는 특수문자 들어간 것 제외
= 알파벳과 숫자로만 시작하는 데이터만 사용하여 더티 데이터 제거


ID, book title은 전부 날린다.





# 알파벳과 숫자의 아스키코드값을 기반으로 추출
selected_data = data[data['Location'].str.replace(r'[^a-zA-Z0-9]', '').astype(bool)]






# 더티 데이터 처리 - null값, 중복값, 미싱 벨류, 


train_df.isnull()




## 특정 값으로 필터링하여 피처 이름 추출
# target_value = 7
# filtered_features = data.columns[data.eq(target_value).any()].tolist()

# 결과 출력
# print(filtered_features)


# 출판사와 저자를 범주형 변수로 변환
# data['Publisher'] = data['Publisher'].astype('category')
# data['Book-Author'] = data['Book-Author'].astype('category')

# 출판사와 저자를 원-핫 인코딩
# encoder = OneHotEncoder(sparse=False)
# encoded_features = pd.DataFrame(encoder.fit_transform(data[['출판사', '저자']]))
# encoded_features.columns = encoder.get_feature_names(['출판사', '저자'])

# 입력 변수(X)와 목표 변수(y) 분리
# X = encoded_features
# y = data['평점']




# 학습 데이터와 테스트 데이터로 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
# model = LinearRegression()
# model.fit(X_train, y_train)

# 테스트 데이터로 평점 예측
# y_pred = model.predict(X_test)

# 예측 결과 출력
# print(y_pred)







# 전체 피쳐 = ID	User-ID	Book-ID	Book-Rating	Age	Location	Book-Title	Book-Author	Year-Of-Publication	Publisher
feature_names = train_df.columns().tolist()

# 피처와 목표 변수 분리
features = ["Publisher", "Book-Author"]
target = ["Book-Rating"]
X = data[features]
y = data[target]

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X, y)

# 회귀 계수 확인
feature_coefs = pd.Series(model.coef_[0], index=features)
sorted_coefs = feature_coefs.abs().sort_values(ascending=False)

# 중요한 피처 2개 선택
selected_features = sorted_coefs[:2].index.tolist()

# 선택된 피처 출력
print(selected_features)





# 알파벳과 숫자의 아스키코드값을 기반으로 추출
selected_data = data[data['Location'].str.replace(r'[^a-zA-Z0-9]', '').astype(bool)]





# 알파벳 대소문자와 숫자인 값만 선택하여 저장
pattern = [a-zA-Z0-9]
selected_data = data[data['Location'].str.match(pattern)]

# 선택된 데이터 출력
print(selected_data)






## 특정 값으로 필터링하여 피처 이름 추출
# target_value = 7
# filtered_features = data.columns[data.eq(target_value).any()].tolist()

# 결과 출력
# print(filtered_features)


# 출판사와 저자를 범주형 변수로 변환
# data['Publisher'] = data['Publisher'].astype('category')
# data['Book-Author'] = data['Book-Author'].astype('category')

# 출판사와 저자를 원-핫 인코딩
encoder = OneHotEncoder(sparse=False)
encoded_features = pd.DataFrame(encoder.fit_transform(data[features]))
encoded_features.columns = encoder.get_feature_names([features])

# 입력 변수(X)와 목표 변수(y) 분리
X = encoded_features
y = data['평점']

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터로 평점 예측
y_pred = model.predict(X_test)

# 예측 결과 출력
print(y_pred)
