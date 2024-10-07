import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_ind, chi2_contingency

# Load the dataset
data = pd.read_csv('Andhra_Health_Data.csv')

# Convert date columns to datetime type
date_columns = ['PREAUTH_DATE', 'CLAIM_DATE', 'SURGERY_DATE', 'DISCHARGE_DATE', 'MORTALITY_DATE']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')

# Data Cleaning
data = data.dropna(subset=['AGE', 'SEX'])

# Feature Engineering
data['HOSPITAL_STAY_DAYS'] = (data['DISCHARGE_DATE'] - data['SURGERY_DATE']).dt.days

# Data Cleaning - Handle missing values
data.fillna({'HOSPITAL_STAY_DAYS': data['HOSPITAL_STAY_DAYS'].mean()}, inplace=True)

# Exploratory Data Analysis (EDA)
# 1. Summary Statistics
print(data.describe())

# 2. Value Counts for Categorical Columns
print(data['SEX'].value_counts(normalize=True))
print(data['CATEGORY_NAME'].value_counts())

# 3. Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 4. Mortality Analysis
mortality_counts = data['Mortality Y / N'].value_counts()
sns.barplot(x=mortality_counts.index, y=mortality_counts.values)
plt.title('Mortality Count')
plt.show()

# 5. Age Distribution
sns.histplot(data['AGE'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()

# 6. Surgery Type Count
sns.countplot(y='SURGERY', data=data)
plt.title('Surgery Type Counts')
plt.show()

# Statistical Testing
# 1. T-test for differences in age between mortality outcomes
age_mortality_yes = data[data['Mortality Y / N'] == 'YES']['AGE']
age_mortality_no = data[data['Mortality Y / N'] == 'NO']['AGE']
ttest_result = ttest_ind(age_mortality_yes, age_mortality_no)
print(f'T-test Result: {ttest_result}')

# 2. Chi-Square Test for associations between categorical variables
crosstab = pd.crosstab(data['SURGERY'], data['Mortality Y / N'])
chi2_result = chi2_contingency(crosstab)
print(f'Chi-Square Result: {chi2_result}')

# Predictive Modeling
# 1. Mortality Prediction - Logistic Regression
data['SEX'] = data['SEX'].map({'Male': 0, 'Female': 1})
X = data[['AGE', 'SEX', 'PREAUTH_AMT']]
y = data['Mortality Y / N'].map({'YES': 1, 'NO': 0})

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Claim Amount Prediction - Random Forest Regressor
X = data[['AGE', 'SEX', 'PREAUTH_AMT']]
y = data['CLAIM_AMOUNT']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

# Clustering Analysis
# 1. KMeans Clustering to group similar patients
kmeans = KMeans(n_clusters=3, random_state=42)
data['CLUSTER'] = kmeans.fit_predict(data[['AGE', 'PREAUTH_AMT', 'CLAIM_AMOUNT']])
sns.scatterplot(x='AGE', y='CLAIM_AMOUNT', hue='CLUSTER', data=data, palette='Set1')
plt.title('KMeans Clustering of Patients')
plt.show()

# Outlier Detection
sns.boxplot(x=data['CLAIM_AMOUNT'])
plt.title('Outlier Detection in Claim Amounts')
plt.show()

# Cost Analysis
# 1. Average Cost per Surgery
avg_cost_per_surgery = data.groupby('SURGERY')['CLAIM_AMOUNT'].mean()
print(avg_cost_per_surgery)

# 2. Cost Analysis by Hospital
avg_cost_per_hospital = data.groupby('HOSP_NAME')['CLAIM_AMOUNT'].mean()
print(avg_cost_per_hospital)

# Geographical Analysis (if geographical information is present)
plt.figure(figsize=(10, 6))
sns.countplot(y='DISTRICT_NAME', data=data, order=data['DISTRICT_NAME'].value_counts().index)
plt.title('District-wise Surgery Distribution')
plt.show()