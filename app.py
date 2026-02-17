import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")
st.title("Tourism Experience Analytics Dashboard")
st.markdown("Predict visit modes, discover personalized attractions, and explore tourism trends!")

# --- LOAD DATA AND MODELS ---
@st.cache_resource
def load_assets():
    # Load models and encoders
    clf_model = joblib.load('models/classification_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    user_item_matrix = joblib.load('models/user_item_matrix.pkl')
    item_similarity_df = joblib.load('models/item_similarity_df.pkl')
    
    # Load dataset for visualizations and mapping
    df = pd.read_csv('data/engineered_tourism_data.csv')
    return clf_model, encoders, user_item_matrix, item_similarity_df, df

clf_model, encoders, user_item_matrix, item_similarity_df, df = load_assets()

# --- CREATE UI TABS ---
tab1, tab2, tab3 = st.tabs(["🔮 Predict Visit Mode", "🎯 Personalized Recommendations", "📊 Tourism Insights"])

# ==========================================
# TAB 1: CLASSIFICATION (Predict Visit Mode)
# ==========================================
with tab1:
    st.header("Predict Your Likely Visit Mode")
    st.write("Enter traveler details below to predict whether this is a Family, Business, or Solo trip.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        continent = st.selectbox("Select Continent", df['UserContinent'].unique())
        region = st.selectbox("Select Region", df[df['UserContinent'] == continent]['UserRegion'].unique())
        country = st.selectbox("Select Country", df[df['UserRegion'] == region]['UserCountry'].unique())
        
    with col2:
        attraction_type = st.selectbox("Preferred Attraction Type", df['AttractionType'].unique())
        visit_month = st.number_input("Visit Month", min_value=0, max_value=12, value=0)
        visit_year = st.selectbox("Visit Year", sorted(df['VisitYear'].unique(), reverse=True))

    if st.button("Predict Visit Mode"):
        try:
            # Encode inputs using the saved label encoders
            cont_enc = encoders['UserContinent'].transform([continent])[0]
            reg_enc = encoders['UserRegion'].transform([region])[0]
            count_enc = encoders['UserCountry'].transform([country])[0]
            attr_enc = encoders['AttractionType'].transform([attraction_type])[0]
            
            # Calculate neutral default values for the missing features
            default_user_rating = df['UserAvgRating'].mean()
            default_attr_rating = df['AttractionAvgRating'].mean()

            # Prepare feature array (Must match the exact order of X_train from Step 4)
            # Features: ['UserContinent_Encoded', 'UserRegion_Encoded', 'UserCountry_Encoded', 'AttractionType_Encoded', 'VisitYear', 'VisitMonth', 'UserAvgRating', 'AttractionAvgRating']
            input_features = [[cont_enc, reg_enc, count_enc, attr_enc, visit_year, visit_month, default_user_rating, default_attr_rating]]
            
            # Predict
            pred_encoded = clf_model.predict(input_features)[0]
            pred_mode = encoders['VisitMode'].inverse_transform([pred_encoded])[0]
            
            # Create a mapping dictionary to translate the ID back to text
            # (Adjust these names if your Mode.xlsx file has different categories)
            mode_mapping = {
                '1': 'Business',
                '2': 'Couples',
                '3': 'Family',
                '4': 'Friends',
                '5': 'Solo'
            }
            
            # Translate the output (fallback to the raw output if not found in dictionary)
            pred_mode_text = mode_mapping.get(str(pred_mode), f"Mode {pred_mode}")
            
            st.success(f"### Predicted Visit Mode: ***{pred_mode_text}***")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}. Ensure all categorical values were seen during training.")

# ==========================================
# TAB 2: RECOMMENDATION SYSTEM
# ==========================================
with tab2:
    st.header("Recommend Attractions for a User")
    st.write("Select an existing User to get collaborative filtering recommendations based on their past ratings.")
    
    # Extract the list of User IDs we want to show
    user_ids_to_show = user_item_matrix.index.tolist()[:100]
    
    # Create a dictionary to map the raw UserId to a nice display string
    user_display_names = {}
    for uid in user_ids_to_show:
        try:
            # Find the user's country from the dataframe to make the label cooler
            country = df[df['UserId'] == uid]['UserCountry'].iloc[0]
            user_display_names[uid] = f"Tourist #{uid} (from {country})"
        except:
            # Fallback just in case the country is missing
            user_display_names[uid] = f"Tourist #{uid}"

    # Use format_func to display the nice names while keeping the actual value as the ID
    user_id = st.selectbox(
        "**Select User**", 
        options=user_ids_to_show,
        format_func=lambda x: user_display_names[x]
    )
    
    # Using a selectbox for a cleaner, more professional look
    num_recs = st.selectbox(
        "**Number of Recommendations to Show**",
        options=[3, 5, 10, 15, 20],
        index=1,  # This sets the default to '5' (the item at index 1)
        format_func=lambda x: f"Top {x} Attractions"
    )
    
    if st.button("Get Recommendations"):
        user_ratings = user_item_matrix.loc[user_id]
        visited_attractions = user_ratings[user_ratings > 0].index.tolist()
        
        sim_scores = item_similarity_df.dot(user_ratings)
        sim_scores = sim_scores.drop(visited_attractions, errors='ignore')
        
        top_attractions_ids = sim_scores.sort_values(ascending=False).head(num_recs).index
        recommendations = df[df['AttractionId'].isin(top_attractions_ids)][['AttractionName', 'AttractionType', 'AttractionCity']].drop_duplicates()
        
        st.write(f"### Top {num_recs} Attractions for {user_display_names[user_id]}:")
        
        # --- Make the result look professional ---
        
        # 1. Reset the index so we don't see those random row numbers
        recommendations = recommendations.reset_index(drop=True)
        
        # 2. Rename columns to look cleaner
        recommendations.columns = ["Attraction Name", "Category", "City"]
        
        # 3. Use st.dataframe with hide_index=True for a modern, interactive look
        st.dataframe(
            recommendations, 
            hide_index=True, 
            width=1000,
        )

# ==========================================
# TAB 3: DATA VISUALIZATIONS & INSIGHTS
# ==========================================
with tab3:
    st.header("Explore Tourism Trends")
    st.write("Visualizations of popular attractions, top regions, and user segments.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Most Popular Attractions")
        top_attractions = df['AttractionName'].value_counts().head(10)
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_attractions.values, y=top_attractions.index, palette='magma', ax=ax1)
        ax1.set_xlabel("Number of Visits")
        st.pyplot(fig1)

    with col2:
        st.subheader("User Segments (Visit Modes)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, y='VisitMode', order=df['VisitMode'].value_counts().index, palette='viridis', ax=ax2)
        ax2.set_xlabel("Count")
        st.pyplot(fig2)
        
    st.subheader("Distribution of Top Regions")
    top_regions = df['UserRegion'].value_counts().head(10)
    fig3, ax3 = plt.subplots(figsize=(5, 2.5)) 
    sns.barplot(x=top_regions.values, y=top_regions.index, palette='crest', ax=ax3)
    ax3.set_xlabel("Number of Visitors")
    
    st.pyplot(fig3, use_container_width=False)

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 Tourism Analytics Inc. | developed by Vadla Shiva Kumar")