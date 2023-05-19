import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




# train.csv 파일 읽기
train_df = pd.read_csv('train.csv')
test_df =  pd.read_csv('test.csv')


# feature
features = ["Book-Author", "Publisher", "User-ID", "Age", "Location"]
target = ["Book-Rating"]


# 피쳐 = 저자, 출판사, 나이, 장소
# 타겟 = 점수

# (1) 데이터의 규모 줄이기
# 1) 쓸모없는 컬럼 제거
# 사용하지 않을 ID, book title은 전부 날린다.
# book title 은 book-ID와 같음, book-ID를 사용, title column자체를 drop
# 
# 2) 더티 데이터, 미싱 데이터의 제거
# 현 데이터에서 미싱 데이터는 없음
# 더티 데이터 제거를 위해 데이터 상관관계 조사
# 
# 분류에 사용 가능한 피쳐
# 유저 아이디 = 장소, 점수, 나이
# 책 아이디 = 저자, 출판사, 책 제목, 발간년도
# 
# 책 제목, 출판사, 발간년도 => 더티 데이터가 많기에 책 아이디로 제목을 대체. 
# 저자 => 북 아이디로 분류한 책 데이터 안에서 저자별로 묶는다?

# 나이의 범위 0~244 => 10세부터 100세로 설정할 것, 나머지 row 드랍

# 장소=> top 35인 location만 사용 
# = 알파벳과 숫자만 들어간 데이터만 사용하여 더티 데이터 제거

# Year=> -1.0 row drop 후 top35만 사용 

장소 드랍, 나이 드랍, 책 제목(column)드랍하고, 







# LabelEncoder를 사용하여 범주형 변수를 숫자로 인코딩
le = LabelEncoder()
for feature in features:
    data[feature] = le.fit_transform(data[feature])

# 피처와 타깃 데이터 분리
X = data[features]
y = data[target]

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = model.predict(X_test)

# 평균 제곱 오차 계산
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

















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




# train.csv 파일 읽기
train_df = pd.read_csv('data/train.csv')



test_df =  pd.read_csv('data/test.csv')


dirty_df = test_df.copy()



clean_df = dirty_df[(dirty_df['Age'] >= 10) & (dirty_df['Age'] <= 100)]


print(len(dirty_df))



print(len(clean_df))


clean_df['Age']






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
