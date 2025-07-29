# 📦 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ✅ Load the dataset
df = pd.read_csv("Unemployment in India.csv")  # Make sure this file is in the same directory

# ✅ Clean column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# ✅ Display first few rows
print("🔍 First few rows:")
print(df.head())

# ✅ Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# ✅ Check for missing values
print("\n🧼 Missing values:")
print(df.isnull().sum())

# ✅ Summary statistics
print("\n📊 Summary statistics:")
print(df.describe())

# ✅ Line plot: Unemployment trend over time
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='date', y='estimated_unemployment_rate_(%)')
plt.title("📉 Unemployment Rate Over Time in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# ✅ COVID-19 impact: March 2020 to Dec 2021
covid_df = df[(df['date'] >= '2020-03-01') & (df['date'] <= '2021-12-31')]

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='date', y='estimated_unemployment_rate_(%)', label="All Data")
if not covid_df.empty:
    sns.lineplot(data=covid_df, x='date', y='estimated_unemployment_rate_(%)', label="COVID-19 Period", color='red')
plt.title("🦠 COVID-19 Impact on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.show()

# ✅ Regional analysis (State-wise)
plt.figure(figsize=(12, 6))
sns.boxplot(x='region', y='estimated_unemployment_rate_(%)', data=df)
plt.title("📌 Regional Unemployment Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ✅ Seasonal trend by month
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

plt.figure(figsize=(10, 5))
sns.boxplot(x='month', y='estimated_unemployment_rate_(%)', data=df)
plt.title("📅 Monthly Seasonal Trend in Unemployment")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# ✅ Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("🧠 Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\n✅ Analysis Complete: Unemployment trends, COVID impact, region & seasonal insights visualized.")