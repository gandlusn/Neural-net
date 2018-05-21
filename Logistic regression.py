import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt;
iris = load_iris();
x = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0
scaler = StandardScaler();
x= scaler.fit_transform(x);#here fit will calculate meand and standard deviation while transofrm will centering and scaling data.
x_tr,x_t,y_tr,y_t = train_test_split(x,y,test_size=0.2,stratify=y)
optimizer = LogisticRegression(C=5,penalty='l1');
optimizer.fit(x_tr,y_tr);
predict = optimizer.predict(x_t)
z_p = accuracy_score(y_t,predict)
print(z_p);
x_rand  = np.linspace(0,3,1000).reshape(-1,1);# linspace(start, end, number of elemets)
y_rand = optimizer.predict_proba(x_rand);
decision = x_rand[y_rand[:,1]>=0.5][0]
plt.plot(x[y==0],y[y==0]);
plt.plot(x[y==1],y[y==1]);
plt.plot([decision,decision],[-1,2])
plt.plot(x_rand, y_rand[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(x_rand, y_rand[:, 0], "b--", linewidth=2, label="Not Iri￼s-Virginica")
plt.xlabel("Petal width")
plt.ylabel("Probability")
plt.text(decision+0.02, 1.5, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.axis([0, 3, -0.02, 1.02])
plt.show();
