#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score


# In[3]:


# Load the dataset
data = pd.read_csv(r"C:\Users\Mouni\Downloads\grad-students.csv")


# In[8]:


data.head(20)


# In[7]:


# Define a dictionary to map long category names to short ones
category_short_names = {
    'Agriculture & Natural Resources': 'Agri & NR',
    'Arts': 'Arts',
    'Biology & Life Science': 'Bio & Life Sci',
    'Business': 'Business',
    'Communications & Journalism': 'Comm & Journ',
    'Computers & Mathematics': 'Comp & Math',
    'Education': 'Education',
    'Engineering': 'Engineering',
    'Health': 'Health',
    'Humanities & Liberal Arts': 'Humanities',
    'Industrial Arts & Consumer Services': 'Ind Arts & Cons',
    'Interdisciplinary': 'Interdisciplinary',
    'Law & Public Policy': 'Law & Policy',
    'Physical Sciences': 'Phys Sci',
    'Psychology & Social Work': 'Psych & Soc Work',
    'Social Science': 'Social Sci'
}

# Replace the long names with short names
data['Major_category'] = data['Major_category'].replace(category_short_names)

# Save the updated dataset to a new file
output_file_path = 'data.csv'
data.to_csv(output_file_path, index=False)

data.head(5)


# In[9]:


# Calculate total graduates and non-graduates
total_grads = data['Grad_total'].sum()
total_nongrads = data['Nongrad_total'].sum()


# In[10]:


# Bar plots for graduates and non-graduates by major category
plt.figure(figsize=(12, 5))

sns.barplot(x='Major_category', y='Grad_total', data=data, color='r')
plt.xticks(rotation=90)
plt.title('Number of Graduates by Major Category')
plt.ylabel('Number of Graduates')


plt.tight_layout()
plt.show()


# In[7]:


# Bar plots for graduates and non-graduates by major category
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
sns.barplot(x='Major_category', y='Nongrad_total', data=data, color='b')
plt.xticks(rotation=90)
plt.title('Number of Non-Graduates by Major Category')
plt.ylabel('Number of Non-Graduates')

plt.tight_layout()
plt.show()


# In[8]:


# Scatter plot for unemployment rate vs median salary (graduates vs non-graduates)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Grad_unemployment_rate', y='Grad_median', data=data, label='Graduates', color='blue')
sns.scatterplot(x='Nongrad_unemployment_rate', y='Nongrad_median', data=data, label='Non-Graduates', color='red')
plt.title('Unemployment Rate vs Median Salary (Graduates vs Non-Graduates)')
plt.xlabel('Unemployment Rate')
plt.ylabel('Median Salary')
plt.legend()
plt.tight_layout()
plt.show()


# In[9]:


# Histogram of Graduate Premium
plt.figure(figsize=(8, 6))
sns.histplot(data['Grad_premium'], kde=True, color='b')
plt.title('Distribution of Graduate Premium')
plt.xlabel('Graduate Premium')
plt.tight_layout()
plt.show()


# In[12]:


# Bar charts for unemployment rate by major category (graduates and non-graduates)
plt.figure(figsize=(10, 5))
sns.barplot(x='Major_category', y='Grad_unemployment_rate', data=data, color='r')
plt.xticks(rotation=90)
plt.title('Graduate Unemployment Rate by Major Category')
plt.ylabel('Unemployment Rate')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='Major_category', y='Nongrad_unemployment_rate', data=data, color='b')
plt.xticks(rotation=90)
plt.title('Non-Graduate Unemployment Rate by Major Category')
plt.ylabel('Unemployment Rate')
plt.tight_layout()
plt.show()


# In[31]:


plt.figure(figsize=(10, 5))

# Graduate full-time year-round employment histogram
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
sns.histplot(data['Grad_full_time_year_round'], kde=True, color='r', bins=20)
plt.title('Distribution of Graduate Full-Time Year-Round Employment', fontsize=14)
plt.xlabel('Graduate Full-Time Year-Round', fontsize=12)
plt.ylabel('Frequency', fontsize=12)


# Show the plots
plt.tight_layout()
plt.show()


# In[32]:


plt.figure(figsize=(10, 5))


# Non-Graduate full-time year-round employment histogram
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
sns.histplot(data['Nongrad_full_time_year_round'], kde=True, color='b', bins=20)
plt.title('Distribution of Non-Graduate Full-Time Year-Round Employment', fontsize=14)
plt.xlabel('Non-Graduate Full-Time Year-Round', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the plots
plt.tight_layout()
plt.show()


# In[14]:


# Correlation heatmap between key variables
plt.figure(figsize=(10, 8))
correlation = data[['Grad_median', 'Nongrad_median', 'Grad_unemployment_rate', 'Nongrad_unemployment_rate', 'Grad_premium']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Key Variables')
plt.tight_layout()
plt.show()


# In[12]:


# Checking for missing values
print(data.isnull().sum())


# In[13]:


# Detecting outliers using IQR and counting them
columns_to_transform = ['Grad_total', 'Grad_sample_size', 'Grad_employed', 'Grad_full_time_year_round', 
                        'Grad_unemployed', 'Grad_unemployment_rate', 'Grad_median', 'Grad_P25', 
                        'Grad_P75', 'Nongrad_total', 'Nongrad_employed', 'Nongrad_full_time_year_round', 
                        'Nongrad_unemployed', 'Nongrad_unemployment_rate', 'Nongrad_median', 
                        'Nongrad_P25', 'Nongrad_P75', 'Grad_share', 'Grad_premium']

def calculate_iqr(data, columns):
    outlier_counts = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    
    return outlier_counts

outlier_counts = calculate_iqr(data, columns_to_transform)
print("Outlier Counts:", outlier_counts)


# In[14]:


# Power transform for normalization
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
data[columns_to_transform] = power_transformer.fit_transform(data[columns_to_transform])


# In[18]:


# Scaling the data
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])


# In[19]:


# Feature engineering: Creating an employment ratio for graduates
data['Grad_employment_ratio'] = data['Grad_employed'] / data['Grad_total']
data.head()


# In[20]:


def calculate_iqr(data, columns):
    outlier_counts = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    
    return outlier_counts

outlier_counts = calculate_iqr(data, columns_to_transform)
print("Outlier Counts:", outlier_counts)


# # Logistic Regression model
# 

# In[21]:


# Logistic Regression model
data['High_unemployment'] = (data['Grad_unemployment_rate'] > data['Grad_unemployment_rate'].median()).astype(int)


# In[22]:


# Select features for logistic regression
features = ['Grad_employed', 'Grad_full_time_year_round', 'Grad_median', 'Grad_P25', 'Grad_P75', 
            'Nongrad_employed', 'Nongrad_full_time_year_round', 'Nongrad_median', 'Nongrad_P25', 'Nongrad_P75']
X = data[features]
y = data['High_unemployment']


# In[23]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[24]:


# Logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)


# In[25]:


# Results for logistic regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))


# In[26]:


# F1, Precision, and Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# In[27]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=['Low Unemployment', 'High Unemployment'], 
            yticklabels=['Low Unemployment', 'High Unemployment'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()




# In[28]:


# AUC Curve
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # Get the probabilities of class 1 (High Unemployment)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)       # Compute False Positive Rate, True Positive Rate
roc_auc = roc_auc_score(y_test, y_pred_proba)       # Calculate AUC score

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='b', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Print AUC score
print("AUC Score:", roc_auc)


# In[29]:


# Check for Overfitting
train_accuracy = log_reg.score(X_train, y_train)
test_accuracy = log_reg.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Cross-validation for Overfitting Check
cross_val_scores = cross_val_score(log_reg, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {cross_val_scores.mean()}")


# In[ ]:




