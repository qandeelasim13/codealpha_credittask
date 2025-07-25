Credit Prediction Project
This project focuses on building a machine learning model to predict whether a customer is creditworthy or not using a complex credit scoring dataset.

📌 Project Overview
The main goal of this project is to develop a classification model that can identify if a customer will be a good or bad credit risk. The workflow includes data loading, preprocessing, encoding, feature scaling, model training, and performance evaluation using standard classification metrics.

📂 Project Steps
1. Data Loading
The dataset is imported from an Excel file named Complex_Credit_Scoring_Dataset.xlsx.

The column Customer_ID is dropped since it does not contribute to prediction.

python
Copy
Edit
df = pd.read_excel("Complex_Credit_Scoring_Dataset.xlsx", sheet_name="Sheet1")
df = df.drop(columns=["Customer_ID"])
2. Data Preprocessing
Label Encoding is used to convert categorical variables (Payment_of_Min_Amount, Payment_Behaviour, etc.) into numeric values.

StandardScaler is used to normalize numerical features for better model performance.

python
Copy
Edit
encoder = LabelEncoder()
df["Payment_of_Min_Amount"] = encoder.fit_transform(df["Payment_of_Min_Amount"])
df["Payment_Behaviour"] = encoder.fit_transform(df["Payment_Behaviour"])
...
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
3. Train-Test Split
The data is divided into training and testing sets using an 80/20 split.

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
4. Model Training
A Random Forest Classifier is used to train the model using the training dataset.

python
Copy
Edit
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
5. Model Evaluation
The model is evaluated using:

Classification Report (Precision, Recall, F1-Score)

ROC-AUC Score (Area Under the Receiver Operating Characteristic Curve)

python
Copy
Edit
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
✅ Output
Trained Model: A Random Forest Classifier capable of predicting creditworthiness.

Metrics Provided:

Precision, Recall, F1-Score (from the classification report)

ROC-AUC Score

🛠️ Technologies Used
Python

Pandas

Scikit-learn

train_test_split

LabelEncoder

StandardScaler

RandomForestClassifier

Jupyter Notebook

📊 Conclusion
This project demonstrates a complete end-to-end credit prediction pipeline using a Random Forest classifier. The model gives reasonable performance and can be further enhanced by:

Feature selection

Hyperparameter tuning

Cross-validation

Trying other algorithms like Gradient Boosting, XGBoost, or Neural Networks

