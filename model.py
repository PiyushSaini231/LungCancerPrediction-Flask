import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection , metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
import pickle
df = pd.read_csv('FeatureSelectionLungDataSet.csv')
df['Level'] = df['Level'].map({'Low':0,'Medium':1, 'High': 2})
X = df.iloc[:,:-1]
y = df['Level']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_train_scaled
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test),columns=X_test.columns)
X_test_scaled
knn = KNeighborsClassifier(n_neighbors=32)  
knn.fit(X_train, y_train)
  # Calculate the accuracy of the model
print(knn.score(X_test, y_test))
#making pickle Object
pickle.dump(knn, open("model.pkl","wb"))
