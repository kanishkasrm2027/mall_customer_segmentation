# 🛍️ Advanced Mall Customer Segmentation Using K-Means Clustering and Principal Component Analysis

An interactive dashboard to segment mall customers with machine learning. This project applies advanced clustering, visual analytics, and actionable recommendations to help data-driven marketing.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mall-customer-segmentation-k-means-pca.streamlit.app/)


## 🚀 Features

-   **Data Exploration**: Instantly view customer data with preprocessing and encoding.
-   **Advanced Segmentation**: K-Means clustering with optimal K selection via Elbow and Silhouette metrics.
-   **Multi-Dimensional Analysis**: Combine age, income, spending score, and gender for deep segmentation.
-   **PCA Visualization**: See clusters projected in 2D space for easy interpretation.
-   **Dynamic Recommendations**: Automated marketing strategies per segment, displayed in expandable cards.
-   **Stylish UI**: Elegant blue dark theme and professional layout.

## 📦 How to Run Locally

### Clone Repository

```bash
git clone https://github.com/kanishkasrm2027/mall_customer_segmentation.git
cd mall_customer_segmentation
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Source

The project uses the `Mall_Customers.csv` dataset. Please ensure it is present in the root directory.

**Credits:** This dataset is provided by the **GeeksforGeeks "21 Days 21 Projects"** course.

### Run Streamlit

```bash
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud

1.  Push code (with `.gitignore` to exclude `.venv/`) to a public GitHub repo.
2.  Go to Streamlit Cloud, click “New app,” and import your repo.
3.  The app uses `requirements.txt` and a custom theme in `.streamlit/config.toml`.

## 📚 About the Project

### Workflow:

-   Data is loaded and processed for clean analysis.
-   Customers are segmented using multi-feature K-Means clustering.
-   Evaluation via visualization (WCSS, Silhouette, PCA).
-   Actionable business recommendations are generated per segment—presented in interactive cards.
-   Users can pick the number of clusters and analyze segment profiles.

### Outcome:

This solution enables targeted marketing by distinguishing customer types such as VIPs, budget spenders, and value-conscious shoppers. All insights are clear and visually appealing for business and academic use.

## 🎨 App Theme

The app uses a blue-accented dark theme for a modern, professional look.
Edit color settings in `.streamlit/config.toml` (primaryColor = "#4B8BBE") as desired.

## 📁 Repository Structure

```text
.
├── app.py                   # Main dashboard code
├── requirements.txt         # All dependencies
├── Mall_Customers.csv       # Example/sample data
├── .gitignore
└── .streamlit/
    └── config.toml          # App theme config
```