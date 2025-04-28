# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load dataset
file_path = r"C:\Users\nihar\OneDrive\Desktop\PYTHON\heart_attack_prediction_indonesia 51.csv"  
df = pd.read_csv(file_path)

# Display basic info and check for missing values
print("Dataset Info:\n")
print(df.info())
print("\nMissing Values:\n")
print(df.isnull().sum())

# Fill numeric missing values with the mean (if any)
df = df.fillna(df.mean(numeric_only=True))

# Ensure column names are consistent and clean
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Debugging: Print column names to verify
print("Columns in the dataset:", df.columns)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# ---------------------------
# 1. Trend Analysis of Heart Attack Cases
# ---------------------------
def trend_analysis(data):
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Handle invalid dates
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month

        # Yearly trend
        yearly_trend = data.groupby('year').size()
        plt.figure(figsize=(10, 6))
        yearly_trend.plot(kind='line', marker='o', title='Yearly Trend of Heart Attack Cases in Indonesia')
        plt.xlabel('Year')
        plt.ylabel('Number of Cases')
        plt.grid()
        plt.show()

        # Monthly trend (if needed)
        monthly_trend = data.groupby(['year', 'month']).size().unstack(fill_value=0)
        plt.figure(figsize=(12, 6))
        sns.heatmap(monthly_trend, cmap='YlGnBu', annot=True, fmt='d')
        plt.title('Monthly Trend of Heart Attack Cases')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.show()
    else:
        print("Date column is missing. Cannot perform trend analysis.")

# ---------------------------
# 2. Geographic Distribution of Heart Attack Cases
# ---------------------------
def geographic_distribution(data):
    if 'region' in data.columns:
        region_distribution = data['region'].value_counts()
        plt.figure(figsize=(12, 6))
        region_distribution.plot(kind='bar', color='skyblue', title='Geographic Distribution of Heart Attack Cases')
        plt.xlabel('Region')
        plt.ylabel('Number of Cases')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()
    else:
        print("Region column is missing. Cannot perform geographic distribution analysis.")

# ---------------------------
# 3. Age and Gender-Based Risk Visualization
# ---------------------------
def age_gender_risk(data):
    if 'age' in data.columns and 'gender' in data.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data, x='age', hue='gender', multiple='stack', kde=False, bins=20, palette='Set2')
        plt.title('Age and Gender-Based Risk Visualization')
        plt.xlabel('Age')
        plt.ylabel('Number of Cases')
        plt.grid(axis='y')
        plt.show()
    else:
        print("Age or Gender column is missing. Cannot perform age and gender-based risk visualization.")

# ---------------------------
# 4. Correlation Between Cholesterol Levels and Heart Attack Risk
# ---------------------------
def cholesterol_correlation(data):
    if 'cholesterol_level' in data.columns and 'age' in data.columns and 'heart_attack' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='cholesterol_level', y='age', hue='heart_attack', palette='coolwarm', alpha=0.7)
        plt.title('Correlation Between Cholesterol Levels and Heart Attack Risk')
        plt.xlabel('Cholesterol Level')
        plt.ylabel('Age')
        plt.grid()
        plt.show()
    else:
        print("Cholesterol Level, Age, or Heart Attack column is missing. Cannot perform correlation analysis.")

# ---------------------------
# 5. Impact of Exercise-Induced Angina on Heart Health
# ---------------------------
def exercise_angina_impact(data):
    if 'exercise_induced_angina' in data.columns and 'age' in data.columns and 'gender' in data.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x='exercise_induced_angina', y='age', hue='gender', palette='Set3')
        plt.title('Impact of Exercise-Induced Angina on Heart Health')
        plt.xlabel('Exercise-Induced Angina')
        plt.ylabel('Age')
        plt.grid(axis='y')
        plt.show()
    else:
        print("Exercise-Induced Angina, Age, or Gender column is missing. Cannot perform analysis.")

# ---------------------------
# 6. Trend Analysis of Resting Blood Pressure Among Patients
# ---------------------------
def resting_bp_trend(data):
    if 'blood_pressure_systolic' in data.columns and 'age' in data.columns and 'heart_attack' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='age', y='blood_pressure_systolic', hue='heart_attack', palette='viridis')
        plt.title('Trend Analysis of Resting Blood Pressure Among Patients')
        plt.xlabel('Age')
        plt.ylabel('Resting Blood Pressure (Systolic)')
        plt.grid()
        plt.show()
    else:
        print("Resting Blood Pressure or Age column is missing. Cannot perform trend analysis.")

# Call the functions to perform the analyses
trend_analysis(df)
geographic_distribution(df)
age_gender_risk(df)
cholesterol_correlation(df)
exercise_angina_impact(df)
resting_bp_trend(df)
