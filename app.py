import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

# ----------------------------------------------------
# Set up Streamlit dashboard and configuration
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="Advanced Mall Customer Segmentation")

# Set consistent plot theme for Matplotlib visuals
sns.set_style('whitegrid')

# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------

@st.cache_data
def load_data():
    """
    Loads Mall Customer data, handles missing file error,
    removes CustomerID, and encodes Gender as binary feature.
    Ensures the app works efficiently with caching.
    """
    try:
        # Load CSV from same directory
        df = pd.read_csv('Mall_Customers.csv')
    except FileNotFoundError:
        st.error("Error: 'Mall_Customers.csv' not found. Please ensure the file is in the same directory as the app.")
        st.stop()
    # Remove CustomerID (not needed for clustering)
    df.drop('CustomerID', axis=1, inplace=True)
    # One-hot encode Gender (binary)
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    df.rename(columns={'Gender_Male': 'Is_Male'}, inplace=True)
    return df

def run_kmeans_analysis(df, features, n_clusters, analysis_name):
    """
    Performs K-Means clustering, computes evaluation metrics,
    and returns cluster assignments and summary statistics.
    """
    X = df[list(features)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    wcss = []
    silhouette_scores = {}
    # Evaluate clustering performance for k = 1 to 10
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        if k >= 2:
            score = silhouette_score(X_scaled, labels)
            silhouette_scores[k] = score
    # Final clustering assignment
    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init='auto')
    df[f'{analysis_name}_Cluster'] = kmeans_model.fit_predict(X_scaled)
    # Compute average profile for each cluster
    profile_cols = [col for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Is_Male'] if col in df.columns]
    cluster_profiles = df.groupby(f'{analysis_name}_Cluster')[profile_cols].mean().round(2)
    cluster_profiles['Size'] = df[f'{analysis_name}_Cluster'].value_counts().sort_index()
    return df, wcss, silhouette_scores, cluster_profiles, X_scaled, scaler

def run_pca_clustering(df, n_clusters):
    """
    Clusters customers using four features, visualizes clusters with PCA,
    and evaluates optimal number of clusters using WCSS and Silhouette.
    """
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Is_Male']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    wcss = []
    silhouette_scores = {}
    # Model selection: k = 1 to 10
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        if k >= 2:
            silhouette_scores[k] = silhouette_score(X_scaled, labels)
    # Fit final K-Means model for selected clusters
    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init='auto')
    labels = kmeans_model.fit_predict(X_scaled)
    df['PCA_Cluster'] = labels
    # Apply PCA for visualizing 4D clusters in 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    pca_df['PCA_Cluster'] = df['PCA_Cluster']
    # Compute cluster mean profiles
    cluster_profiles = df.groupby('PCA_Cluster')[features].mean().round(2)
    cluster_profiles['Size'] = df['PCA_Cluster'].value_counts().sort_index()
    return df, wcss, silhouette_scores, cluster_profiles, pca_df, pca.explained_variance_ratio_

def plot_evaluation(wcss, silhouette_scores, title):
    """
    Plots WCSS (Elbow) and Silhouette Score to help select optimal cluster count.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(range(1, 11), wcss, marker='o', linestyle='--')
    axes[0].set_title('Elbow Method (WCSS)')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('WCSS (Inertia)')
    axes[0].set_xticks(range(1, 11))
    axes[0].grid(True)
    ks_sil = list(silhouette_scores.keys())
    scores_sil = list(silhouette_scores.values())
    axes[1].plot(ks_sil, scores_sil, marker='o', linestyle='--', color='red')
    axes[1].set_title('Silhouette Score (Higher is Better)')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_xticks(ks_sil)
    axes[1].grid(True)
    fig.suptitle(title, fontsize=16)
    return fig

def plot_profile_bars(profiles, title):
    """
    Plots cluster-wise bar charts for average Age, Income, and Spending Score.
    """
    features_to_plot = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, metric in enumerate(features_to_plot):
        profiles[metric].plot(kind='bar', ax=axes[i], rot=0, color=sns.color_palette("tab10", len(profiles)))
        axes[i].set_title(f'Mean {metric}')
        axes[i].set_xlabel('Cluster ID')
    fig.suptitle(f'Cluster Profile Comparison: {title}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def generate_recommendations(profiles_df, original_df):
    """
    Interprets cluster characteristics and generates marketing recommendations
    and descriptive names for each customer segment.
    """
    global_mean = original_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Is_Male']].mean()
    recommendations = {}
    for cluster_id in profiles_df.index:
        profile = profiles_df.loc[cluster_id]
        flags = {
            'Age': 'Young' if profile['Age'] < global_mean['Age'] * 0.9 else 'Older',
            'Income': 'High Income' if profile['Annual Income (k$)'] > global_mean['Annual Income (k$)'] * 1.1 else 'Low/Mid Income',
            'Spending': 'High Spender' if profile['Spending Score (1-100)'] > global_mean['Spending Score (1-100)'] * 1.1 else 'Low/Mid Spender',
            'Gender': 'Male Dominated' if profile['Is_Male'] > 0.55 else ('Female Dominated' if profile['Is_Male'] < 0.45 else 'Balanced')
        }
        name_parts = []
        if flags['Spending'] == 'High Spender' and flags['Income'] == 'High Income':
            name_parts.append('Top-Tier VIPs')
        elif flags['Spending'] == 'High Spender':
            name_parts.append('Budget Spenders')
        elif flags['Spending'] == 'Low/Mid Spender' and flags['Income'] == 'High Income':
            name_parts.append('Needs-Based')
        else:
            name_parts.append('Value-Conscious')
        name = f"{name_parts[0]} ({flags['Age']} {flags['Gender']})"
        # Attach recommendation based on segment type
        rec = ""
        if name_parts[0] == 'Top-Tier VIPs':
            rec = "**Retention & Premium:** Offer exclusive early access, dedicated personal shopping services, and premium loyalty point multipliers to maximize lifetime value."
        elif name_parts[0] == 'Budget Spenders':
            rec = "**Incentivize & Convert:** Utilize mobile flash sales, 'buy now pay later' options, and gift card promotions to drive impulse purchases."
        elif name_parts[0] == 'Needs-Based':
            rec = "**Engagement & Discovery:** Send highly personalized, curated emails suggesting relevant new products. Offer free experience add-ons (e.g., free coffee/lounge access)."
        else: # Value-Conscious
            rec = "**Focus on Essentials:** Highlight discounts on bulk purchases, guarantee the lowest prices on staples, and use geo-fencing for timely in-store coupons."
        recommendations[cluster_id] = {'Cluster Name': name, 'Recommendation': rec.strip()}
    return pd.DataFrame.from_dict(recommendations, orient='index')

# ----------------------------------------------------
# Streamlit App Layout Starts Here
# ----------------------------------------------------

st.title("🛍️ Advanced Mall Customer Segmentation Project")
st.markdown("""
This project analyzes mall customer data using advanced segmentation techniques. 
It applies multi-dimensional K-Means clustering, evaluates performance with Elbow and Silhouette metrics, 
visualizes clusters with PCA, and presents actionable marketing strategies for each segment.
""")

# Sidebar controls for user-defined cluster numbers
st.sidebar.header("Segmentation Controls")
income_n_clusters = st.sidebar.slider('K for Income-Spending (2D)', 2, 8, 5)
age_n_clusters = st.sidebar.slider('K for Age-Spending (2D)', 2, 8, 4)
pca_n_clusters = st.sidebar.slider('K for Combined 4D Analysis', 2, 8, 5)
st.sidebar.markdown("---")

# Load cleaned data and prepare working copies for each analysis
original_df = load_data()
df_income = original_df.copy()
df_age = original_df.copy()
df_pca = original_df.copy()

# Section 1: Data Overview, structure, and statistics
st.header("1. Data Overview & Preprocessing")
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("**First 5 Rows (Encoded):**")
    st.dataframe(original_df.head(), height=200)
    st.write(f"Data shape: {original_df.shape}")
    st.write("`Is_Male` is the One-Hot Encoded version of Gender used for clustering.")
with col2:
    st.markdown("**Descriptive Statistics:**")
    st.dataframe(original_df.describe().T)

# -- Sections 2 & 3 for 2D K-Means omitted for brevity; retain original code for those --

# Section 4: Combined 4D Clustering and PCA Visualization
st.header("4. Combined 4D Segmentation (Age, Income, Spending, Gender)")
st.markdown("Customers are segmented based on their age, income, spending score, and gender for a comprehensive analysis. Principal Component Analysis (PCA) helps visualize the high-dimensional clusters in 2D.")

df_pca, wcss3, silhouette_scores3, profiles3, pca_df, explained_variance = run_pca_clustering(
    df_pca,
    pca_n_clusters
)

st.subheader("4.1 Model Evaluation")
fig_eval3 = plot_evaluation(wcss3, silhouette_scores3, 'K-Means Evaluation for 4D Combined Data')
st.pyplot(fig_eval3)

st.subheader("4.2 Visualization via Principal Component Analysis (PCA)")
st.markdown(f"PCA explained **{explained_variance.sum()*100:.2f}%** of the total variance in the first two components.")
fig_pca = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='PCA_Cluster',
    title=f'4D Customer Segments Projected onto 2 PCA Components (K={pca_n_clusters})',
    hover_data={'PC1': ':.2f', 'PC2': ':.2f', 'PCA_Cluster': True}
)
st.plotly_chart(fig_pca)

st.subheader("4.3 Final Cluster Profiles")
col7, col8 = st.columns(2)
with col7:
    st.markdown("**Cluster Profile Bar Comparison**")
    fig_bars3 = plot_profile_bars(profiles3, "4D Combined")
    st.pyplot(fig_bars3)
with col8:
    st.markdown(f"**Cluster Profiles (K={pca_n_clusters}):**")
    st.dataframe(profiles3)

# Section 5: Actionable Strategies for Each Segment
st.header("5. Final Actionable Insights & Marketing Strategy 🎯")
final_recommendations_df = generate_recommendations(profiles3, original_df)
final_report_df = pd.concat([profiles3, final_recommendations_df], axis=1)
st.markdown("""
Each cluster is analyzed and assigned a descriptive segment name along with a targeted marketing recommendation.
Explore each segment below for actionable business strategies tailored to its profile.
""")
st.subheader("Automated Cluster Names and Marketing Recommendations")

# --- NEW: Expanded cards for each cluster ---
for cluster_id, row in final_report_df.iterrows():
    with st.expander(f"Cluster {cluster_id}: {row['Cluster Name']} (Size: {row['Size']})"):
        st.markdown(f"""
        - **Average Age:** {row['Age']}
        - **Average Annual Income (k$):** {row['Annual Income (k$)']}
        - **Average Spending Score:** {row['Spending Score (1-100)']}
        - **Proportion Male:** {row['Is_Male']:.2f}
        ---
        **Recommendation:**  
        {row['Recommendation']}
        """)

st.markdown("""
---
*All recommendations are tailored to each cluster’s demographic and behavioral profile, providing actionable strategies for targeted marketing campaigns and resource allocation.*
""")
# --- Project Summary Section ---
st.markdown("""
<hr>
<h3>📝 Project Summary: Mall Customer Segmentation</h3>

<ul>
  <li><b>📚 Data Loaded:</b> Demographic and behavioral data of mall customers, including Age, Income, Spending, and Gender.</li>
  <li><b>🔬 Data Processing:</b> Applied normalization, handled categorical data, and ensured readiness for machine learning clustering.</li>
  <li><b>🧠 Clustering:</b> Used advanced <i>K-Means</i> to segment customers by multiple features and evaluated cluster quality with Elbow and Silhouette metrics.</li>
  <li><b>🖼️ Visualization:</b> Presented interactive cluster plots, PCA-based scatterplots, and comparative bar charts for easy interpretation.</li>
  <li><b>🚀 Recommendations:</b> Automatically generated, segment-specific marketing strategies for each cluster with clear segment naming.</li>
</ul>

<b>Outcome:</b>
<br>
Segmented customers into actionable groups—empowering data-driven decision-making and enabling the design of targeted, effective marketing campaigns for each unique shopper type.
""", unsafe_allow_html=True)

