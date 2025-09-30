# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
     import pandas as pd
     df1=pd.read_csv("C:\\Users\\admin\\Downloads\\bmi.csv")
     df1
```

<img width="469" height="449" alt="image" src="https://github.com/user-attachments/assets/b1eef647-9585-4ee7-9052-fe924ee8b2f5" />

```
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer,RobustScaler

df2=df1.copy()
enc=StandardScaler()
df2[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df2

```

<img width="594" height="443" alt="image" src="https://github.com/user-attachments/assets/ddf1a74f-5e7f-40d6-a6f2-9bbc7d6d790f" />


```
df3=df1.copy()
enc=MinMaxScaler()
df3[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df3
```

<img width="642" height="443" alt="image" src="https://github.com/user-attachments/assets/8fd05b84-c59c-4e14-8dcf-3c37fa132f80" />

```
df4=df1.copy()
enc=MaxAbsScaler()
df4[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df4

```

<img width="581" height="436" alt="image" src="https://github.com/user-attachments/assets/a99eefe3-2aaa-47a8-8185-1edf01fc6a5d" />

```
df5=df1.copy()
enc=Normalizer()
df5[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df5

```
<img width="606" height="443" alt="image" src="https://github.com/user-attachments/assets/2898058c-d023-4d89-bfd4-677b7a12a512" />

```

df6=df1.copy()
enc=RobustScaler()
df6[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df6

```
<img width="682" height="443" alt="image" src="https://github.com/user-attachments/assets/d4de6f76-9a84-474a-b508-b7a82e3652b0" />

```
import pandas as pd

df=pd.read_csv("C:\\Users\\admin\\Downloads\\income(1) (1).csv")
df
```
<img width="1243" height="755" alt="image" src="https://github.com/user-attachments/assets/5980069f-3748-496a-b6bd-f210aaffe546" />

```
from sklearn.preprocessing import LabelEncoder

df_encoded=df.copy()
le=LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
x=df_encoded.drop("SalStat",axis=1)
y=df_encoded["SalStat"]

```

```
x

```
<img width="1234" height="438" alt="image" src="https://github.com/user-attachments/assets/4e6ff755-0298-4c75-8f2a-ab563a683210" />

```
y

```

<img width="520" height="276" alt="image" src="https://github.com/user-attachments/assets/68c82ad7-c55b-43a6-b963-7517baa45b82" />

```
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2

chi2_selector=SelectKBest(chi2,k=5)
chi2_selector.fit(x,y)

selected_features_chi2=x.columns[chi2_selector.get_support()]
print("Selected features(chi_square):",list(selected_features_chi2))

mi_scores=pd.Series(chi2_selector.scores_,index = x.columns)
print(mi_scores.sort_values(ascending=False))

```

<img width="1199" height="315" alt="image" src="https://github.com/user-attachments/assets/6f40035f-d024-4066-89ba-c56394d9e995" />

```
from sklearn.feature_selection import f_classif

anova_selector=SelectKBest(f_classif,k=5)
anova_selector.fit(x,y)

selected_features_anova=x.columns[anova_selector.get_support()]
print("Selected features(chi_square):",list(selected_features_anova))

mi_scores=pd.Series(anova_selector.scores_,index = x.columns)
print(mi_scores.sort_values(ascending=False))

```

<img width="1133" height="319" alt="image" src="https://github.com/user-attachments/assets/fdf56de6-045c-40cc-a7b2-109aec799936" />

```
from sklearn.feature_selection import mutual_info_classif

mi_selector=SelectKBest(mutual_info_classif,k=5)
mi_selector.fit(x,y)

selected_features_mi=x.columns[mi_selector.get_support()]
print("Selected features(Mutual Info):",list(selected_features_mi))

mi_scores=pd.Series(anova_selector.scores_,index = x.columns)
print(mi_scores.sort_values(ascending=False))

```


<img width="1136" height="307" alt="image" src="https://github.com/user-attachments/assets/3bbaf865-9892-4a8d-802c-0228e2e534f5" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(max_iter=100)
rfe= RFE(model, n_features_to_select=5)
rfe.fit(x,y)

selected_features_rfe = x.columns[rfe.support_]
print("Selected features (RFE):",list(selected_features_rfe))

```

<img width="864" height="48" alt="image" src="https://github.com/user-attachments/assets/e2f823df-8a82-45e9-b313-c30e2c749669" />


```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

model = LogisticRegression(max_iter=100)
rfe= SequentialFeatureSelector(model, n_features_to_select=5)
rfe.fit(x,y)

selected_features_rfe = x.columns[rfe.get_support()]
print("Selected features (SF):",list(selected_features_rfe))

```

<img width="986" height="43" alt="image" src="https://github.com/user-attachments/assets/114e4445-c9f4-4411-a745-dd02d9663aff" />

```
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x,y)

importances = pd.Series(rf.feature_importances_, index=x.columns)
selected_features_rf = importances.sort_values(ascending=False).head(5).index
print(importances)
print("Top 5 features (Random Forest Importance):",list(selected_features_rf))

```


<img width="1211" height="310" alt="image" src="https://github.com/user-attachments/assets/cb0fa961-fdb6-4535-935f-ac03ea723b8f" />


```
from sklearn.linear_model import LassoCV
import numpy as np
 
    
lasso=LassoCV(cv=5).fit(x,y)
importance = np.abs(lasso.coef_)

selected_features_lasso=x.columns[importance>0]
print("Selected features (Lasso):",list(selected_features_lasso))

```

<img width="861" height="50" alt="image" src="https://github.com/user-attachments/assets/8863f258-e7fa-451b-86c7-83b5f891bfc3" />

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("C:\\Users\\admin\\Downloads\\income(1) (1).csv")

df_encoded = df.copy()

le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("SalStat", axis=1)
y = df_encoded["SalStat"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)  # you can tune k
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

```


<img width="722" height="362" alt="image" src="https://github.com/user-attachments/assets/7ff928b2-bc5f-4e46-9397-7dbac22ae701" />



# RESULT:

   Thus, Feature selection and Feature scaling has been used on thegiven dataset.

