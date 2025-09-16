import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Bengaluru House Price Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('bengaluru_house_prices_cleaned.csv')
        return df
    except FileNotFoundError:
        st.error("Please make sure 'bengaluru_house_prices_cleaned.csv' is in the same directory")
        return None


# Data preprocessing for ML
@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning"""
    # Create a copy
    data = df.copy()

    # Handle missing values
    data = data.dropna()

    # Extract numeric values from size column
    data['bhk'] = data['size'].str.extract('(\d+)').astype(float)

    # Convert price to numeric (assuming it's in lakhs)
    data['price_numeric'] = pd.to_numeric(data['price'], errors='coerce')

    # Create location groups for top locations
    top_locations = data['location'].value_counts().head(20).index
    data['location_grouped'] = data['location'].apply(
        lambda x: x if x in top_locations else 'Other'
    )

    # Encode categorical variables
    le_location = LabelEncoder()
    le_area_type = LabelEncoder()
    le_availability = LabelEncoder()

    data['location_encoded'] = le_location.fit_transform(data['location_grouped'])
    data['area_type_encoded'] = le_area_type.fit_transform(data['area_type'])
    data['availability_encoded'] = le_availability.fit_transform(data['availability'])

    return data, le_location, le_area_type, le_availability


# Main app
def main():
    st.markdown('<h1 class="main-header">🏠 Bengaluru House Price Analyzer</h1>',
                unsafe_allow_html=True)

    # Load data
    df = load_data()
    if df is None:
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Explorer", "📈 Visualizations", "🤖 Price Predictor", "📋 Insights"]
    )

    if page == "🏠 Home":
        home_page(df)
    elif page == "📊 Data Explorer":
        data_explorer_page(df)
    elif page == "📈 Visualizations":
        visualizations_page(df)
    elif page == "🤖 Price Predictor":
        price_predictor_page(df)
    elif page == "📋 Insights":
        insights_page(df)


def home_page(df):
    """Home page with overview"""
    st.markdown('<h2 class="sub-header">Welcome to Bengaluru House Price Analyzer!</h2>',
                unsafe_allow_html=True)

    st.write("""
    This app helps you explore and analyze house prices in Bengaluru. You can:
    - 📊 Explore the dataset with interactive filters
    - 📈 View beautiful visualizations and trends
    - 🤖 Predict house prices using ML models
    - 📋 Get insights about the real estate market
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        st.metric("Unique Locations", f"{df['location'].nunique()}")
    with col3:
        avg_price = df['price'].mean()
        st.metric("Average Price", f"₹{avg_price:.2f} L")
    with col4:
        max_price = df['price'].max()
        st.metric("Highest Price", f"₹{max_price:.2f} L")

    # Sample data preview
    st.markdown('<h3 class="sub-header">Dataset Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)


def data_explorer_page(df):
    """Data exploration page with filters"""
    st.markdown('<h2 class="sub-header">📊 Data Explorer</h2>', unsafe_allow_html=True)

    # Filters in sidebar
    st.sidebar.markdown("### Filters")

    # Location filter
    locations = ['All'] + sorted(df['location'].unique().tolist())
    selected_location = st.sidebar.selectbox("Select Location", locations)

    # BHK filter
    bhk_options = ['All'] + sorted(df['size'].unique().tolist())
    selected_bhk = st.sidebar.selectbox("Select BHK", bhk_options)

    # Price range filter
    min_price, max_price = st.sidebar.slider(
        "Price Range (in Lakhs)",
        min_value=float(df['price'].min()),
        max_value=float(df['price'].max()),
        value=(float(df['price'].min()), float(df['price'].max()))
    )

    # Area type filter
    area_types = ['All'] + df['area_type'].unique().tolist()
    selected_area_type = st.sidebar.selectbox("Select Area Type", area_types)

    # Apply filters
    filtered_df = df.copy()

    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['location'] == selected_location]

    if selected_bhk != 'All':
        filtered_df = filtered_df[filtered_df['size'] == selected_bhk]

    if selected_area_type != 'All':
        filtered_df = filtered_df[filtered_df['area_type'] == selected_area_type]

    filtered_df = filtered_df[
        (filtered_df['price'] >= min_price) &
        (filtered_df['price'] <= max_price)
        ]

    # Display filtered results
    st.write(f"**Showing {len(filtered_df)} properties matching your filters**")

    if len(filtered_df) > 0:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Price", f"₹{filtered_df['price'].mean():.2f} L")
        with col2:
            st.metric("Average Area", f"{filtered_df['total_sqft'].mean():.0f} sqft")
        with col3:
            st.metric("Average Price/sqft", f"₹{filtered_df['price_per_sqft'].mean():.0f}")

        # Display data
        st.dataframe(filtered_df, use_container_width=True)

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name="filtered_house_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No properties match your current filters.")


def visualizations_page(df):
    """Visualizations page"""
    st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)

    # Chart type selection
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Price Distribution", "Location Analysis", "BHK Analysis", "Area vs Price", "Price Trends"]
    )

    if chart_type == "Price Distribution":
        st.subheader("Price Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig_hist = px.histogram(
                df, x='price', nbins=50,
                title="Price Distribution",
                labels={'price': 'Price (Lakhs)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Box plot
            fig_box = px.box(
                df, y='price',
                title="Price Box Plot",
                labels={'price': 'Price (Lakhs)'}
            )
            st.plotly_chart(fig_box, use_container_width=True)

    elif chart_type == "Location Analysis":
        st.subheader("Location Analysis")

        # Top 15 locations by average price
        top_locations = df.groupby('location')['price'].agg(['mean', 'count']).reset_index()
        top_locations = top_locations[top_locations['count'] >= 5].nlargest(15, 'mean')

        fig_loc = px.bar(
            top_locations, x='location', y='mean',
            title="Top 15 Locations by Average Price",
            labels={'mean': 'Average Price (Lakhs)', 'location': 'Location'}
        )
        fig_loc.update_xaxes(tickangle=45)
        st.plotly_chart(fig_loc, use_container_width=True)

        # Price by location scatter plot
        sample_locations = df['location'].value_counts().head(10).index
        filtered_data = df[df['location'].isin(sample_locations)]

        fig_scatter = px.scatter(
            filtered_data, x='total_sqft', y='price', color='location',
            title="Price vs Area by Top Locations",
            labels={'total_sqft': 'Total Area (sqft)', 'price': 'Price (Lakhs)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif chart_type == "BHK Analysis":
        st.subheader("BHK Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # BHK distribution
            bhk_counts = df['size'].value_counts()
            fig_bhk = px.pie(
                values=bhk_counts.values, names=bhk_counts.index,
                title="Distribution of BHK Types"
            )
            st.plotly_chart(fig_bhk, use_container_width=True)

        with col2:
            # Price by BHK
            bhk_price = df.groupby('size')['price'].mean().sort_values(ascending=False)
            fig_bhk_price = px.bar(
                x=bhk_price.index, y=bhk_price.values,
                title="Average Price by BHK Type",
                labels={'x': 'BHK Type', 'y': 'Average Price (Lakhs)'}
            )
            st.plotly_chart(fig_bhk_price, use_container_width=True)

    elif chart_type == "Area vs Price":
        st.subheader("Area vs Price Analysis")

        # Scatter plot with trend line
        fig_scatter = px.scatter(
            df.sample(1000), x='total_sqft', y='price',
            trendline="ols",
            title="Area vs Price Relationship",
            labels={'total_sqft': 'Total Area (sqft)', 'price': 'Price (Lakhs)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Price per sqft analysis
        fig_psqft = px.histogram(
            df, x='price_per_sqft', nbins=50,
            title="Price per Sqft Distribution",
            labels={'price_per_sqft': 'Price per Sqft (₹)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_psqft, use_container_width=True)

    elif chart_type == "Price Trends":
        st.subheader("Price Trends by Different Factors")

        # Availability analysis
        fig_avail = px.box(
            df, x='availability', y='price',
            title="Price Distribution by Availability",
            labels={'availability': 'Availability', 'price': 'Price (Lakhs)'}
        )
        fig_avail.update_xaxes(tickangle=45)
        st.plotly_chart(fig_avail, use_container_width=True)

        # Area type analysis
        fig_area_type = px.violin(
            df, x='area_type', y='price',
            title="Price Distribution by Area Type",
            labels={'area_type': 'Area Type', 'price': 'Price (Lakhs)'}
        )
        st.plotly_chart(fig_area_type, use_container_width=True)


def price_predictor_page(df):
    """Price prediction page using ML models"""
    st.markdown('<h2 class="sub-header">🤖 Price Predictor</h2>', unsafe_allow_html=True)

    st.write("Enter property details to predict the price using machine learning:")

    # Preprocess data
    try:
        processed_df, le_location, le_area_type, le_availability = preprocess_data(df)

        # Input form
        col1, col2 = st.columns(2)

        with col1:
            # Location selection
            location_options = sorted(df['location'].unique())
            selected_location = st.selectbox("Location", location_options)

            # BHK selection
            bhk_options = sorted([int(x.split()[0]) for x in df['size'].unique() if x.split()[0].isdigit()])
            selected_bhk = st.selectbox("BHK", bhk_options)

            # Total area
            total_sqft = st.number_input("Total Area (sqft)", min_value=300, max_value=10000, value=1000)

        with col2:
            # Area type
            area_type_options = df['area_type'].unique()
            selected_area_type = st.selectbox("Area Type", area_type_options)

            # Bathrooms
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

            # Balconies
            balconies = st.number_input("Balconies", min_value=0, max_value=5, value=1)

        # Prediction button
        if st.button("🔮 Predict Price", type="primary"):
            # Prepare features for prediction
            try:
                # Create feature vector
                location_encoded = le_location.transform([selected_location])[
                    0] if selected_location in le_location.classes_ else 0
                area_type_encoded = le_area_type.transform([selected_area_type])[0]
                availability_encoded = 0  # Default for 'Ready To Move'

                # Prepare training data
                features = ['location_encoded', 'area_type_encoded', 'availability_encoded',
                            'total_sqft', 'bath', 'balcony', 'bhk']

                X = processed_df[features].dropna()
                y = processed_df['price_numeric'].dropna()

                # Align X and y
                common_idx = X.index.intersection(y.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]

                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Random Forest Model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)

                # Linear Regression Model
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)

                # Make predictions
                input_features = np.array([[location_encoded, area_type_encoded, availability_encoded,
                                            total_sqft, bathrooms, balconies, selected_bhk]])

                rf_prediction = rf_model.predict(input_features)[0]
                lr_prediction = lr_model.predict(input_features)[0]

                # Display predictions
                st.success("✅ Price Prediction Complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Random Forest Prediction", f"₹{rf_prediction:.2f} L")
                with col2:
                    st.metric("Linear Regression Prediction", f"₹{lr_prediction:.2f} L")
                with col3:
                    avg_pred = (rf_prediction + lr_prediction) / 2
                    st.metric("Average Prediction", f"₹{avg_pred:.2f} L")

                # Model performance
                st.subheader("Model Performance")

                rf_test_pred = rf_model.predict(X_test)
                lr_test_pred = lr_model.predict(X_test)

                performance_data = {
                    'Model': ['Random Forest', 'Linear Regression'],
                    'R² Score': [r2_score(y_test, rf_test_pred), r2_score(y_test, lr_test_pred)],
                    'MAE': [mean_absolute_error(y_test, rf_test_pred), mean_absolute_error(y_test, lr_test_pred)]
                }

                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, use_container_width=True)

                # Feature importance (Random Forest)
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig_importance = px.bar(
                    feature_importance, x='Feature', y='Importance',
                    title="Feature Importance (Random Forest)"
                )
                st.plotly_chart(fig_importance, use_container_width=True)

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

    except Exception as e:
        st.error(f"Error in data preprocessing: {str(e)}")


def insights_page(df):
    """Insights and analytics page"""
    st.markdown('<h2 class="sub-header">📋 Market Insights</h2>', unsafe_allow_html=True)

    # Key insights
    st.subheader("🔍 Key Market Insights")

    # Top insights
    insights = [
        f"📍 **Most Expensive Area:** {df.groupby('location')['price'].mean().idxmax()} with avg price ₹{df.groupby('location')['price'].mean().max():.2f} L",
        f"🏠 **Most Common BHK:** {df['size'].value_counts().index[0]} ({df['size'].value_counts().iloc[0]} properties)",
        f"💰 **Price Range:** ₹{df['price'].min():.2f} L - ₹{df['price'].max():.2f} L",
        f"📊 **Average Price per Sqft:** ₹{df['price_per_sqft'].mean():.0f}",
        f"🏗️ **Most Common Area Type:** {df['area_type'].value_counts().index[0]}"
    ]

    for insight in insights:
        st.markdown(f"- {insight}")

    st.divider()

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Top 10 Locations by Count")
        top_locations_count = df['location'].value_counts().head(10)
        fig_count = px.bar(
            x=top_locations_count.values, y=top_locations_count.index,
            orientation='h',
            title="Properties Count by Location"
        )
        fig_count.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_count, use_container_width=True)

    with col2:
        st.subheader("💎 Premium vs Budget Areas")

        # Categorize areas
        area_stats = df.groupby('location')['price_per_sqft'].agg(['mean', 'count']).reset_index()
        area_stats = area_stats[area_stats['count'] >= 5]  # At least 5 properties

        premium_threshold = area_stats['mean'].quantile(0.8)
        budget_threshold = area_stats['mean'].quantile(0.2)

        premium_areas = area_stats[area_stats['mean'] >= premium_threshold]['location'].tolist()
        budget_areas = area_stats[area_stats['mean'] <= budget_threshold]['location'].tolist()

        st.write("**🏆 Premium Areas (Top 20%):**")
        for area in premium_areas[:5]:
            price = area_stats[area_stats['location'] == area]['mean'].iloc[0]
            st.write(f"• {area}: ₹{price:.0f}/sqft")

        st.write("**💰 Budget-Friendly Areas (Bottom 20%):**")
        for area in budget_areas[:5]:
            price = area_stats[area_stats['location'] == area]['mean'].iloc[0]
            st.write(f"• {area}: ₹{price:.0f}/sqft")

    st.divider()

    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")

    # Select numeric columns for correlation
    numeric_cols = ['price', 'total_sqft', 'bath', 'balcony', 'price_per_sqft']
    corr_matrix = df[numeric_cols].corr()

    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=numeric_cols,
        y=numeric_cols,
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)


if __name__ == "__main__":
    main()
