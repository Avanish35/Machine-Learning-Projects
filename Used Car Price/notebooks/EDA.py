import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"E:\Projects\Used Car Price\data\used_cars.csv")

df.head()
df.info()
df.describe()

plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50)
plt.title("Car Price Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='manufacturing_year', y='price', data=df)
plt.title("Price vs Manufacturing Year")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='fuel_type', y='price', data=df)
plt.title("Fuel Type vs Price")
plt.show()
