import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
df = pd.read_csv('bengaluru_house_prices.csv')

print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ================== BASIC DATA EXPLORATION ==================
print("\n" + "="*50)
print("BASIC DATA EXPLORATION")
print("="*50)

# Check data types
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicated rows
print(f"\nDuplicated Rows: {df.duplicated().sum()}")

# ================== DATA CLEANING STEPS ==================
print("\n" + "="*50)
print("DATA CLEANING PROCESS")
print("="*50)

# Step 1: Remove unnecessary columns (if any)
print("\n1. Checking for unnecessary columns...")
print("All columns seem relevant for house price prediction")

# Step 2: Handle missing values
print("\n2. Handling missing values...")

# Check missing percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("Missing value percentages:")
for col, percent in missing_percentage.items():
    if percent > 0:
        print(f"{col}: {percent:.2f}%")

# Handle missing values in each column
# Society: Fill with 'Unknown' as it's categorical
df['society'].fillna('Unknown', inplace=True)

# Bath: Fill with median based on size
df['bath'] = df.groupby('size')['bath'].transform(lambda x: x.fillna(x.median()))

# Balcony: Fill with 0 (assuming no balcony if not mentioned)
df['balcony'].fillna(0, inplace=True)

# Location: Drop rows with missing location (very few)
df = df.dropna(subset=['location'])

# Size: Drop rows with missing size (crucial feature)
df = df.dropna(subset=['size'])

print("Missing values after handling:")
print(df.isnull().sum())

# Step 3: Clean and standardize 'size' column
print("\n3. Cleaning size column...")

# Check unique values in size column
print("Unique values in size column:")
print(df['size'].value_counts().head(10))

# Function to extract number of bedrooms/rooms
def extract_bhk(size_str):
    if pd.isna(size_str):
        return np.nan
    
    size_str = str(size_str).strip()
    
    # Handle BHK format
    if 'BHK' in size_str:
        return int(size_str.split(' ')[0])
    
    # Handle Bedroom format
    elif 'Bedroom' in size_str:
        return int(size_str.split(' ')[0])
    
    # Handle RK format (Room + Kitchen)
    elif 'RK' in size_str:
        return int(size_str.split(' ')[0])
    
    # Handle other formats
    else:
        # Try to extract first number
        import re
        numbers = re.findall(r'\d+', size_str)
        if numbers:
            return int(numbers[0])
        else:
            return np.nan

# Apply the function
df['bhk'] = df['size'].apply(extract_bhk)

# Check for any remaining NaN values in bhk
print(f"Missing values in BHK after extraction: {df['bhk'].isnull().sum()}")

# Drop rows where BHK extraction failed
df = df.dropna(subset=['bhk'])

# Step 4: Clean 'total_sqft' column
print("\n4. Cleaning total_sqft column...")

# Check data types and formats
print("Sample total_sqft values:")
print(df['total_sqft'].head(10))

def clean_sqft(sqft_str):
    if pd.isna(sqft_str):
        return np.nan
    
    sqft_str = str(sqft_str).strip()
    
    # Handle range values (e.g., "1200 - 1300")
    if '-' in sqft_str:
        values = sqft_str.split('-')
        try:
            return (float(values[0]) + float(values[1])) / 2
        except:
            return np.nan
    
    # Handle other non-numeric formats
    try:
        # Remove any non-numeric characters except decimal point
        import re
        cleaned = re.sub(r'[^\d.]', '', sqft_str)
        if cleaned:
            return float(cleaned)
        else:
            return np.nan
    except:
        return np.nan

# Apply cleaning function
df['total_sqft_cleaned'] = df['total_sqft'].apply(clean_sqft)

# Check results
print(f"Missing values in cleaned sqft: {df['total_sqft_cleaned'].isnull().sum()}")

# Drop rows with invalid sqft values
df = df.dropna(subset=['total_sqft_cleaned'])

# Step 5: Remove outliers
print("\n5. Removing outliers...")

# Remove houses with unrealistic sqft per room
df['sqft_per_room'] = df['total_sqft_cleaned'] / df['bhk']

# Remove houses with sqft per room less than 200 (too small)
df = df[df['sqft_per_room'] >= 200]

# Remove price outliers using IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Price bounds: Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")

# Remove extreme price outliers
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# Step 6: Standardize categorical variables
print("\n6. Standardizing categorical variables...")

# Clean location names (remove extra spaces, standardize case)
df['location'] = df['location'].str.strip().str.title()

# Clean area_type
df['area_type'] = df['area_type'].str.strip()

# Clean availability
df['availability'] = df['availability'].str.strip()

# Step 7: Feature Engineering
print("\n7. Feature engineering...")

# Create price per sqft
df['price_per_sqft'] = df['price'] / df['total_sqft_cleaned'] * 100000  # Convert to price per sqft

# Categorize locations by frequency
location_counts = df['location'].value_counts()
locations_to_keep = location_counts[location_counts >= 10].index
df['location_category'] = df['location'].apply(
    lambda x: x if x in locations_to_keep else 'Other'
)

# Step 8: Final cleanup
print("\n8. Final cleanup...")

# Select final columns
final_columns = [
    'area_type', 'availability', 'location', 'location_category', 
    'size', 'bhk', 'society', 'total_sqft_cleaned', 'bath', 
    'balcony', 'price', 'sqft_per_room', 'price_per_sqft'
]

df_cleaned = df[final_columns].copy()

# Rename columns for clarity
df_cleaned.rename(columns={
    'total_sqft_cleaned': 'total_sqft',
    'location_category': 'location_grouped'
}, inplace=True)

# Reset index
df_cleaned.reset_index(drop=True, inplace=True)

# ================== FINAL SUMMARY ==================
print("\n" + "="*50)
print("CLEANING SUMMARY")
print("="*50)

print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df_cleaned.shape}")
print(f"Rows removed: {len(df) - len(df_cleaned)}")
print(f"Percentage of data retained: {(len(df_cleaned)/len(df)*100):.2f}%")

print("\nFinal dataset info:")
print(df_cleaned.info())

print("\nFinal missing values:")
print(df_cleaned.isnull().sum())

print("\nFirst 5 rows of cleaned data:")
print(df_cleaned.head())

# ================== SAVE CLEANED DATA ==================
print("\n9. Saving cleaned data...")

# Save to CSV
df_cleaned.to_csv('bengaluru_house_prices_cleaned.csv', index=False)
print("Cleaned data saved as 'bengaluru_house_prices_cleaned.csv'")

# ================== OPTIONAL: DATA VISUALIZATION ==================
print("\n10. Creating basic visualizations...")

# Set up plotting
plt.figure(figsize=(15, 12))

# Price distribution
plt.subplot(2, 3, 1)
plt.hist(df_cleaned['price'], bins=50, edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price (Lakhs)')
plt.ylabel('Frequency')

# BHK distribution
plt.subplot(2, 3, 2)
df_cleaned['bhk'].value_counts().sort_index().plot(kind='bar')
plt.title('BHK Distribution')
plt.xlabel('Number of BHK')
plt.ylabel('Count')

# Area type distribution
plt.subplot(2, 3, 3)
df_cleaned['area_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Area Type Distribution')

# Price vs Total Sqft
plt.subplot(2, 3, 4)
plt.scatter(df_cleaned['total_sqft'], df_cleaned['price'], alpha=0.5)
plt.title('Price vs Total Sqft')
plt.xlabel('Total Sqft')
plt.ylabel('Price (Lakhs)')

# Price per sqft distribution
plt.subplot(2, 3, 5)
plt.hist(df_cleaned['price_per_sqft'], bins=50, edgecolor='black')
plt.title('Price per Sqft Distribution')
plt.xlabel('Price per Sqft')
plt.ylabel('Frequency')

# Top 10 locations
plt.subplot(2, 3, 6)
top_locations = df_cleaned['location_grouped'].value_counts().head(10)
top_locations.plot(kind='barh')
plt.title('Top 10 Locations by Count')
plt.xlabel('Count')

plt.tight_layout()
plt.savefig('data_cleaning_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nData cleaning completed successfully!")
print("Check 'bengaluru_house_prices_cleaned.csv' for the cleaned dataset")
print("Check 'data_cleaning_visualizations.png' for basic visualizations")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('bengaluru_house_prices_cleaned.csv')

print("Dataset Info:")
print(data.info())
print("\nDataset Shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Basic data exploration
print("\nMissing values:")
print(data.isnull().sum())
print("\nBasic statistics:")
print(data.describe())

# Data Preprocessing
def preprocess_data(df):
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Handle missing values
    df_processed.fillna(df_processed.mean(numeric_only=True), inplace=True)
    df_processed.fillna('Unknown', inplace=True)
    
    # Feature engineering
    # Extract numeric values from size column if it contains BHK info
    df_processed['size_numeric'] = df_processed['size'].str.extract('(\d+)').astype(float)
    
    # Create price per sqft ratio feature
    df_processed['price_per_sqft_ratio'] = df_processed['price'] / df_processed['total_sqft']
    
    return df_processed

# Preprocess the data
data_processed = preprocess_data(data)

# Define features and target
# Remove price_per_sqft as it's derived from target variable to avoid data leakage
features_to_drop = ['price', 'price_per_sqft', 'sqft_per_room']
X = data_processed.drop(features_to_drop, axis=1)
y = data_processed['price']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify categorical and numerical columns
categorical_features = ['area_type', 'availability', 'location', 'location_grouped', 'size', 'society']
numerical_features = ['bhk', 'total_sqft', 'bath', 'balcony', 'size_numeric', 'price_per_sqft_ratio']

# Create preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Define models to train
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    
    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std(),
        'model': pipeline
    }
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Display results summary
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[model]['MAE'] for model in results],
    'RMSE': [results[model]['RMSE'] for model in results],
    'R² Score': [results[model]['R2'] for model in results],
    'CV R² Mean': [results[model]['CV_R2_mean'] for model in results]
})

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(results_df.sort_values('R² Score', ascending=False).to_string(index=False))

# Hyperparameter tuning for the best model
best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

if best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }
else:
    param_grid = {}

if param_grid:
    base_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', models[best_model_name])
    ])
    
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate the tuned model
    tuned_pred = grid_search.predict(X_test)
    tuned_r2 = r2_score(y_test, tuned_pred)
    tuned_mae = mean_absolute_error(y_test, tuned_pred)
    tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
    
    print(f"Tuned model performance:")
    print(f"R² Score: {tuned_r2:.4f}")
    print(f"MAE: {tuned_mae:.2f}")
    print(f"RMSE: {tuned_rmse:.2f}")

# Feature importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Get feature names after preprocessing
    feature_names = (numerical_features + 
                    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    
    # Get feature importance
    best_model = results[best_model_name]['model']
    feature_importance = best_model.named_steps['model'].feature_importances_
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Important Features for {best_model_name}:")
    print(importance_df.head(10).to_string(index=False))

# Visualization
plt.figure(figsize=(15, 10))

# Model comparison
plt.subplot(2, 3, 1)
models_names = list(results.keys())
r2_scores = [results[model]['R2'] for model in models_names]
plt.bar(models_names, r2_scores)
plt.title('R² Score Comparison')
plt.xticks(rotation=45)
plt.ylabel('R² Score')

# Actual vs Predicted for best model
plt.subplot(2, 3, 2)
best_pred = results[best_model_name]['model'].predict(X_test)
plt.scatter(y_test, best_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted - {best_model_name}')

# Residual plot
plt.subplot(2, 3, 3)
residuals = y_test - best_pred
plt.scatter(best_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Feature importance plot (if applicable)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    plt.subplot(2, 3, 4)
    top_features = importance_df.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')

# Price distribution
plt.subplot(2, 3, 5)
plt.hist(y, bins=50, alpha=0.7)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution')

# Model performance metrics
plt.subplot(2, 3, 6)
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [results[best_model_name]['MAE'], 
              results[best_model_name]['RMSE'], 
              results[best_model_name]['R2']]
})
plt.bar(metrics_df['Metric'], metrics_df['Value'])
plt.title(f'{best_model_name} Performance Metrics')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Save the best model
import joblib
joblib.dump(results[best_model_name]['model'], 'best_house_price_model.pkl')
print(f"\nBest model ({best_model_name}) saved as 'best_house_price_model.pkl'")

# Function to make predictions on new data
def predict_price(model, area_type, availability, location, location_grouped, 
                 size, bhk, society, total_sqft, bath, balcony):
    """
    Function to predict house price for new data
    """
    new_data = pd.DataFrame({
        'area_type': [area_type],
        'availability': [availability],
        'location': [location],
        'location_grouped': [location_grouped],
        'size': [size],
        'bhk': [bhk],
        'society': [society],
        'total_sqft': [total_sqft],
        'bath': [bath],
        'balcony': [balcony],
        'size_numeric': [bhk],  # Assuming BHK matches size_numeric
        'price_per_sqft_ratio': [0]  # Will be calculated later
    })
    
    prediction = model.predict(new_data)
    return prediction[0]

print("\nModel training completed successfully!")
print(f"Best performing model: {best_model_name}")
print(f"Best R² Score: {results[best_model_name]['R2']:.4f}")
