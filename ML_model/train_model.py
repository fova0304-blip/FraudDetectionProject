from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import joblib

# describe()로 확인후 결측치가 없지만 outlier는 있는걸 확인함-> Amount, V27, V28
# 하지만 fraud 같은경우는 정상 vs 비정상거래가 극단적인 차이가 있으니 이상치를 걍 두는게 좋음
# 결측치 없으니 그냥 pass
df = pd.read_csv("creditcard.csv")
X = df.drop("Class",axis=1)
y = df["Class"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

'''
왜 randomforest?:
이번 fraud 데이터는 대규모이며 고차원 tabular 구조를 가짐
rf 원리: bagging -> variance감소 -> 과적합 줄어들음
트리 앙상블 -> 대규모·고차원 tabular 데이터에 안정적으로 적용 가능함
대신 근본적인 데이터누수, 클래스 불균형은 따로 신경써줘야함
'''

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)

rf.fit(X_train,y_train)
pred = rf.predict(X_test)
#threshold adjustment
pred_proba = rf.predict_proba(X_test)[:,1]
pred_proba_45 =  rf.predict_proba(X_test)[:,1] > 0.45
pred_proba_45 = [1 if proba == True else 0 for proba in pred_proba_45 ]


# Accuracy만 하기엔 평가 지표가 너무 약함
# Fraud는 조금 위험성만 있어도 잡아내야하기떄문에 confusion matrix가 필수임
print(f"accuracy: {rf.score(X_test,y_test)}")
print(f"roc_auc_score: {roc_auc_score(y_test, pred_proba)}")
print(f"recall_score: {recall_score(y_test, pred)},threshold=0.5")
print(f"recall_score: {recall_score(y_test,pred_proba_45)},threshold=0.45")
print(f"confusion_matrix: {confusion_matrix(y_test, pred)}")

'''
accuracy: 0.9995611109160493
roc_auc_score: 0.9703826997174785
recall_score: 0.7857142857142857
confusion_matrix: [[56860     4]
                   [   21    77]]
'''

'''
Fraud에서는 recall이 제일 중요함 진짜 사기인걸 정확히 골라내야하니깐
그러므로 디폴트가 0.5(50%)인 threshold는 부족할수도있음 
그래서 threshold는 조절하는게 맞음 예) 0.5 -> 0.45
'''


os.makedirs("model",exist_ok=True)
joblib.dump(rf,"model/train_model.joblib")
