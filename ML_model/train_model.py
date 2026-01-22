from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)

rf.fit(X_train,y_train)
rf.score(X_test,y_test) #0.99 이상

os.makedirs("model",exist_ok=True)
joblib.dump(rf,"model/train_model.joblib")
