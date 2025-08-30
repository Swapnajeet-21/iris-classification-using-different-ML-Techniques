import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
knn = KNeighborsClassifier(n_neighbors=5)
svc = SVC(probability=True, kernel="linear")
log_reg = LogisticRegression(max_iter=200)

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
log_reg.fit(X_train, y_train)

# Streamlit UI
st.title("🌸 Iris Flower Classification")
st.write("Compare predictions using **KNN**, **SVC**, and **Logistic Regression**")

# Sidebar inputs
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:,0].min()), float(X[:,0].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X[:,1].min()), float(X[:,1].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:,2].min()), float(X[:,2].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X[:,3].min()), float(X[:,3].max()))

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_data = scaler.transform(input_data)

# Predictions
pred_knn = iris.target_names[knn.predict(input_data)[0]]
pred_svc = iris.target_names[svc.predict(input_data)[0]]
pred_log = iris.target_names[log_reg.predict(input_data)[0]]

# Show results
st.subheader("🔮 Predictions")
st.write(f"**KNN:** {pred_knn}")
st.write(f"**SVC:** {pred_svc}")
st.write(f"**Logistic Regression:** {pred_log}")

# Accuracy comparison
st.subheader("📊 Model Accuracy on Test Data")
st.write(f"KNN Accuracy: {knn.score(X_test, y_test):.2f}")
st.write(f"SVC Accuracy: {svc.score(X_test, y_test):.2f}")
st.write(f"Logistic Regression Accuracy: {log_reg.score(X_test, y_test):.2f}")
