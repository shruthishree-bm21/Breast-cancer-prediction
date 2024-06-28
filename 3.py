import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the breast cancer dataset from scikit-learn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Streamlit app
st.title('Breast Cancer Prediction')
st.subheader('Enter Patient Details')

# User input for feature values
input_data = {}
for feature_name in data.feature_names:
    value = st.number_input(f'{feature_name}', step=0.01)
    input_data[feature_name] = value

# Create a dataframe from user input
input_df = pd.DataFrame([input_data])

# Scale the user input
input_scaled = scaler.transform(input_df)

# Make predictions
prediction = model.predict(input_scaled)
prediction_prob = model.predict_proba(input_scaled)

# Display prediction
st.subheader('Prediction')
if prediction[0] == 1:
    st.write('The tumor is predicted to be begnin (non-cancerous).')
else:
    st.write('The tumor is predicted to be malignant (cancerous).')

# Display prediction probabilities
st.subheader('Prediction Probabilities')
proba_df = pd.DataFrame({'Malignant': prediction_prob[0][0], 'Begnin': prediction_prob[0][1]}, index=[0])
st.dataframe(proba_df.style.format('{:.2%}'))

# Confusion matrix
st.subheader('Confusion Matrix')
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Display confusion matrix
fig_cm = plt.gcf()
st.pyplot(fig_cm)

# Classification report
st.subheader('Classification Report')
classification_rep = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
st.text(classification_rep)

# Feature importance
st.subheader('Feature Importance')
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig_imp = plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Top Feature Importance')

# Display feature importance
st.pyplot(fig_imp)
