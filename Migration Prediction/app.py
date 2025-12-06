import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

st.set_page_config(page_title="Migration NZ Explorer", layout="wide")

@st.cache_data
def load_data(path="migration_nz.csv"):
    df = pd.read_csv(path)
    return df

@st.cache_data
def preprocess(df):
    # copy to avoid mutating original cached df
    data = df.copy()
    # Replace Measure text with numeric categories (same as your notebook)
    data['Measure'].replace({"Arrivals":0, "Departures":1, "Net":2}, inplace=True)

    # Handle missing values in "Value"
    if "Value" in data.columns:
        data["Value"].fillna(data["Value"].median(), inplace=True)

    # Create factorized IDs for Country and Citizenship
    data['CountryID'], country_index = pd.factorize(data['Country'])
    data['CitID'], cit_index = pd.factorize(data['Citizenship'])

    # Keep the mappings to allow later lookup / reverse mapping
    country_map = dict(enumerate(country_index))
    cit_map = dict(enumerate(cit_index))

    return data, country_map, cit_map

@st.cache_resource
def train_model(data):
    # Features and label
    X = data[['CountryID','Measure','Year','CitID']].values
    y = data['Value'].values

    # train-test split (though we will use final model trained on all data for predictions)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

    rf = RandomForestRegressor(n_estimators=70, max_features=3, max_depth=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    # compute a simple metric for display
    preds = rf.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return rf, mse

# --- UI ---
st.title("🗺️ Migration NZ — Explorer & Predictor")
st.write("Query historical migration and get model predictions for a chosen Country / Citizenship / Measure / Year.")

# Load and preprocess
data_load_state = st.text("Loading data...")
df_raw = load_data()
data, country_map, cit_map = preprocess(df_raw)
data_load_state.text("Data loaded ✅")

# Sidebar controls
st.sidebar.header("Query options")
unique_countries = list(data['Country'].dropna().unique())
unique_cits = list(data['CitID'].map(lambda id: cit_map[id]))
measures = {0:"Arrivals", 1:"Departures", 2:"Net"}

sel_country = st.sidebar.selectbox("Select Country (destination)", unique_countries)
# get possible citizenships for selected country (if any)
available_cits = data.loc[data['Country'] == sel_country, 'Citizenship'].unique()
sel_citizen = st.sidebar.selectbox("Select Citizenship (origin)", available_cits)
sel_measure_text = st.sidebar.selectbox("Select Measure", ["Arrivals","Departures","Net"])
# convert measure to numeric for model
measure_num = {"Arrivals":0,"Departures":1,"Net":2}[sel_measure_text]

min_year = int(data['Year'].min())
max_year = int(data['Year'].max())
sel_year = st.sidebar.slider("Select Year", min_year, max_year+5, value=max_year)  # allow predicting into near future

# train model (cached)
with st.spinner("Training model (cached) ..."):
    model, mse = train_model(data)

st.sidebar.markdown(f"**Model MSE (on test split):** {mse:.2f}")

# Filter historical rows
hist = data[(data['Country'] == sel_country) & (data['Citizenship'] == sel_citizen) & (data['Measure'] == measure_num)]
hist_sorted = hist.sort_values('Year')

st.subheader(f"Historical data: {sel_country} ← {sel_citizen} ({sel_measure_text})")
if hist_sorted.empty:
    st.write("No historical records found for this exact pair. Showing nearest matches.")
    # try looser match: same country and measure
    hist_sorted = data[(data['Country'] == sel_country) & (data['Measure'] == measure_num)].sort_values('Year')
    if hist_sorted.empty:
        st.write("Also no records for same country + measure. Try a different combination.")
        st.stop()

# Show top 10 rows
st.dataframe(hist_sorted[['Year','Value']].reset_index(drop=True).tail(20))

# Plot time-series
fig, ax = plt.subplots(figsize=(8,3))
sns.lineplot(data=hist_sorted, x='Year', y='Value', marker='o', ax=ax)
ax.set_title(f"{sel_country} ← {sel_citizen} — {sel_measure_text} (Value over years)")
ax.set_ylabel("Value")
ax.set_xlabel("Year")
st.pyplot(fig)

# Prepare input for model prediction
# We need the numeric IDs used during preprocessing
country_id = int(pd.factorize(data['Country'])[0][data.index[data['Country'] == sel_country][0]]) \
    if sel_country in list(data['Country']) else data.loc[data['Country']==sel_country, 'CountryID'].iloc[0]

# Simpler: get the factorized id directly from preprocessed table (safer)
try:
    country_id = int(data.loc[data['Country']==sel_country, 'CountryID'].iloc[0])
except Exception:
    country_id = 0
try:
    cit_id = int(data.loc[data['Citizenship']==sel_citizen, 'CitID'].iloc[0])
except Exception:
    cit_id = 0

X_query = np.array([[country_id, measure_num, int(sel_year), cit_id]])

pred_value = model.predict(X_query)[0]

st.subheader("Model prediction")
st.write(f"Predicted `{sel_measure_text}` value for **{sel_country} ← {sel_citizen}** in **{sel_year}** :")
st.metric(label="Predicted Value", value=f"{pred_value:.0f}")

# Show predicted vs historical if year already exists
if sel_year in hist_sorted['Year'].values:
    actual = hist_sorted.loc[hist_sorted['Year']==sel_year, 'Value'].values[0]
    st.write(f"Actual value for {sel_year} (from dataset): {actual}")
    st.write(f"Absolute error: {abs(actual - pred_value):.2f}")

# Quick summary: Top countries for a chosen year
st.subheader("Top destination countries by total migration for a chosen year")
chosen_year_for_summary = st.slider("Choose year to rank (summary)", min_year, max_year, value=max_year, key="summary_year")
summary = data[data['Year'] == chosen_year_for_summary].groupby('Country').agg({'Value':'sum'}).sort_values('Value', ascending=False).reset_index()
st.table(summary.head(10).assign(Rank=lambda d: range(1, len(d)+1)).set_index('Rank'))

# Download filtered historical data
csv = hist_sorted.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download historical data (CSV)",
    data=csv,
    file_name=f"migration_{sel_country}_{sel_citizen}_{sel_measure_text}.csv",
    mime='text/csv'
)

# Footer / notes
st.markdown("---")
st.markdown("""
**Notes & tips**
- The model is a basic RandomForest trained on the dataset's numeric encodings — it's a quick predictor for demo purposes.
- For production you should:  
  1) ensure factorization mappings are persisted and reused,  
  2) train on more features, perform time-based cross validation, and tune hyperparameters,  
  3) validate predictions against held-out future years.
""")
