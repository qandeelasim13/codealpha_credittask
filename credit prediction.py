#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# # data loading

# In[3]:


df = pd.read_excel("Complex_Credit_Scoring_Dataset.xlsx", sheet_name="Sheet1")
df = df.drop(columns=["Customer_ID"])
df.head(30)


# In[5]:


df.tail()


# In[7]:


df.shape


# In[9]:


df.sample(5)


# In[10]:


df.info()


# In[11]:


df.describe()


# # CHECKING NULL VALUE

# In[12]:


df.isnull().sum()


#  # Feature engineering

# In[13]:


df["Debt_to_Income"] = df["Debts"] / df["Income"]


# In[17]:


import pandas as pd
import numpy as np


# # Safe drop for Customer_ID column

# In[15]:


if 'Customer_ID' in df.columns:
    df.drop(columns=["Customer_ID"], inplace=True)


# # Separate numeric and non-numeric columns

# In[18]:


numeric_cols = df.select_dtypes(include=[np.number])
non_numeric_cols = df.select_dtypes(exclude=[np.number])


# # Fill missing values in numeric columns

# In[19]:


numeric_cols.fillna(numeric_cols.mean(), inplace=True)


# # Combine numeric and non-numeric columns back together

# In[21]:


df_cleaned = pd.concat([numeric_cols, non_numeric_cols], axis=1)


# # Preview cleaned data

# In[22]:


print(df_cleaned.head())


# # Separate columns by type

# In[23]:


numeric_cols = df.select_dtypes(include=[np.number])
non_numeric_cols = df.select_dtypes(exclude=[np.number])


# # Fill missing values in numeric columns with mean

# In[24]:


numeric_cols.fillna(numeric_cols.mean(), inplace=True)


# # Fill missing values in non-numeric columns with mode

# In[25]:


for col in non_numeric_cols.columns:
    non_numeric_cols[col].fillna(non_numeric_cols[col].mode()[0], inplace=True)


# # Recombine into one cleaned DataFrame

# In[26]:


data = pd.concat([numeric_cols, non_numeric_cols], axis=1)


# In[27]:


print(data.head())


# # DROP MISSING VALUES

# # Separate columns by type

# In[28]:


numeric_cols = data.select_dtypes(include=[np.number])
non_numeric_cols = data.select_dtypes(exclude=[np.number])


# # Fill missing values

# In[29]:


numeric_cols.fillna(numeric_cols.mean(), inplace=True)
for col in non_numeric_cols.columns:
    non_numeric_cols[col].fillna(non_numeric_cols[col].mode()[0], inplace=True)


# # Combine back

# In[30]:


data = pd.concat([numeric_cols, non_numeric_cols], axis=1)


# # Final safety step: drop any remaining rows with NaNs

# In[31]:


data.dropna(inplace=True)


# # Print any remaining missing values (should be 0s)

# In[32]:


missing_values = data.isnull().sum()
print("Missing values after cleaning:\n", missing_values)


# # CHECKING SHAPE OF DATA

# In[33]:


data.shape


# # CHECKING OUTLIERS

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Select numeric columns
numeric_cols = data.select_dtypes(include=[np.number])

# Step 2: Visualize boxplot before outlier removal
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
numeric_cols.boxplot()
plt.title("Before Outlier Removal")

# Step 3: Calculate IQR and remove outliers
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Apply the IQR filter row-wise
data_cleaned = data[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 4: Visualize boxplot after outlier removal
plt.subplot(1, 2, 2)
data_cleaned.select_dtypes(include=[np.number]).boxplot()
plt.title("After Outlier Removal")

plt.tight_layout()
plt.show()


# # DATA TRANSFORMATION

# In[35]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=[np.number])
non_numeric_cols = data.select_dtypes(exclude=[np.number])

# Initialize and apply MinMaxScaler
scaler = MinMaxScaler()
scaled_numeric_data = scaler.fit_transform(numeric_cols)

# Convert scaled data back to DataFrame
scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols.columns)

# Reset index to align rows before concatenation
non_numeric_cols.reset_index(drop=True, inplace=True)

# Combine scaled numeric data with original categorical columns
scaled_data = pd.concat([scaled_numeric_df, non_numeric_cols], axis=1)

# Output the result
print(f"Scaled data shape: {scaled_data.shape}")
print("\n" + "*" * 60)
print(scaled_data.head())


# # STANDARIZATION
# 

# In[36]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_excel("Complex_Credit_Scoring_Dataset.xlsx", sheet_name="Sheet1")

# Drop 'Customer_ID' if it exists
if 'Customer_ID' in df.columns:
    df.drop(columns=['Customer_ID'], inplace=True)

# Separate numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number])
non_numeric_cols = df.select_dtypes(exclude=[np.number])

# Fill missing values
numeric_cols.fillna(numeric_cols.mean(), inplace=True)
for col in non_numeric_cols.columns:
    non_numeric_cols[col].fillna(non_numeric_cols[col].mode()[0], inplace=True)

# Scale numeric features
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_cols)
scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_cols.columns)

# Combine numeric + categorical
non_numeric_cols.reset_index(drop=True, inplace=True)
scaled_data = pd.concat([scaled_numeric_df, non_numeric_cols], axis=1)

# Label Encode categorical columns
label_enc = LabelEncoder()
for col in scaled_data.select_dtypes(exclude=[np.number]).columns:
    scaled_data[col] = label_enc.fit_transform(scaled_data[col])

# Final shape and preview
print(f"Scaled data shape: {scaled_data.shape}")
print("*" * 60)
print(scaled_data.head())


# # Train-Test Split and Standardization

# In[37]:


categorical_cols = ["Payment_History", "Loan_Purpose", "Employment_Status", "Marital_Status"]
df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)


# In[38]:


X = df.drop(columns=["Defaulted"])
y = df["Defaulted"]


# In[39]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[41]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# # Predictions and probability scores

# In[42]:


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

# FIX: Use the original (unscaled) target from df
y = df["Defaulted"]  # unscaled original target
X = scaled_data.drop(columns=["Defaulted"])  # all features are scaled

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Check class distribution
print("Before balancing:", Counter(y_train))
print("After balancing:", Counter(y_train_bal))

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_bal, y_train_bal)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))


# # Artificial Neural Network (ANN)

# In[45]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Target and Features
y = df["Defaulted"]  # use original, unscaled target
X = scaled_data.drop(columns=["Defaulted"])  # scaled features

# Encode y if it's not already numeric
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Before balancing:", Counter(y_train))
print("After balancing:", Counter(y_train_bal))

# ANN Model
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_bal, y_train_bal, 
                    validation_split=0.2,
                    epochs=100, batch_size=16, callbacks=[early_stop], verbose=1)

# Predict
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# # evaluate model

# In[50]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict on test set
y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")

# Classification Report
from sklearn.metrics import classification_report
print("ANN Classification Report:\n")
print(classification_report(y_test, y_pred_ann))


# # confusion matrix

# In[51]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict on test data
y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_ann)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad", "Good"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - ANN")
plt.show()


# # random forest

# In[72]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_excel("C:/Users/ideal/Downloads/Complex_Credit_Scoring_Dataset.xlsx")

# 2. Drop missing values
data.dropna(inplace=True)

# 3. Set target column
target_column = "Defaulted"  # Change made here

# 4. Drop non-feature columns like ID
data = data.drop(columns=["Customer_ID"])

# 5. Encode categorical features
categorical_cols = data.select_dtypes(include="object").columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# 6. Split features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 8. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Predict and Evaluate
y_pred = model.predict(X_test)
print(" Classification Report:\n", classification_report(y_test, y_pred))
print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Plot Feature Importances
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], color="skyblue", align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=45, ha="right")
plt.tight_layout()
plt.show()



# In[73]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict
y_pred = model.predict(X_test)

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


# # Model Comparison

# In[74]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))


# In[ ]:




