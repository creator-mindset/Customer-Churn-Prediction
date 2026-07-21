

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🛍️",
    layout="wide",
)

CATEGORICAL_OPTIONS = {
    "favorite_category": ["Accessories", "Beauty", "Electronics", "Fashion",
                           "Footwear", "Home Decor", "Sports"],
    "customer_country": ["Australia", "Canada", "Germany", "India",
                          "UAE", "UK", "USA"],
    "favorite_traffic_source": ["Direct", "Email", "Organic",
                                 "Paid Ads", "Social Media"],
    "favorite_payment_method": ["Apple Pay", "Cash on Delivery",
                                 "Credit Card", "Debit Card", "PayPal"],
}

FRIENDLY_LABELS = {
    "favorite_category": "Favorite product category",
    "customer_country": "Country",
    "favorite_traffic_source": "How they usually find the store",
    "favorite_payment_method": "Preferred payment method",
    "frequency": "Number of past orders",
    "tenure_days": "Days since their first order",
    "avg_order_value": "Average order value ($)",
    "avg_discount_percent": "Average discount used (%)",
    "avg_rating": "Average product rating given (1-5)",
    "return_rate": "Share of orders returned (0-1)",
    "avg_shipping_cost": "Average shipping cost ($)",
    "avg_quantity": "Average items per order",
    "recency_days": "Days since their last order",
}


# ------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return {
        "clf": joblib.load("churn_classifier.pkl"),
        "reg": joblib.load("revenue_regressor.pkl"),
        "kmeans": joblib.load("kmeans_model.pkl"),
        "cluster_scaler": joblib.load("cluster_scaler.pkl"),
        "encoders": joblib.load("label_encoders.pkl"),
        "segment_name_map": joblib.load("segment_name_map.pkl"),
        "clf_features": joblib.load("clf_features.pkl"),
        "reg_features": joblib.load("reg_features.pkl"),
        "cluster_features": joblib.load("cluster_features.pkl"),
    }


@st.cache_data
def load_data():
    return pd.read_csv("customer_features_scored.csv")


artifacts = load_artifacts()
data = load_data()
encoders = artifacts["encoders"]
segment_name_map = artifacts["segment_name_map"]
data["segment_name"] = data["segment"].map(segment_name_map)


def encode_row(row_dict: dict) -> dict:
    """Turn human-readable category strings into the label-encoded
    integers the models were trained on."""
    encoded = dict(row_dict)
    for col, le in encoders.items():
        if col in encoded:
            encoded[col] = le.transform([encoded[col]])[0]
    return encoded


# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("🛍️ Churn Intelligence")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🔮 Predict Churn", "💰 Predict Customer Value",
     "🧩 Customer Segments"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on order-level Shopify data, aggregated to one row per "
    "customer (RFM features), with a Random Forest classifier, a "
    "Random Forest regressor, and KMeans clustering."
)

# ------------------------------------------------------------------
# PAGE 1: OVERVIEW
# ------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("Shopify Customer Churn & Value Intelligence")
    st.write(
        "This app predicts **who is likely to churn**, **how much a "
        "customer is worth**, and **which segment they belong to** — "
        "using three machine learning models trained on your historical "
        "order data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(data):,}")
    c2.metric("Churn rate", f"{data['churn'].mean()*100:.1f}%")
    c3.metric("Avg. customer revenue", f"${data['total_revenue'].mean():,.0f}")
    c4.metric("Avg. orders / customer", f"{data['frequency'].mean():.1f}")

    st.markdown("### Churn breakdown")
    fig1 = px.pie(
        data, names=data["churn"].map({0: "Active", 1: "Churned"}),
        title="Active vs Churned Customers", hole=0.45,
        color_discrete_sequence=["#2E8B57", "#DC143C"],
    )
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.box(
            data, x="segment_name", y="total_revenue", color="segment_name",
            title="Revenue distribution by segment",
            labels={"segment_name": "Segment", "total_revenue": "Total revenue ($)"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        cat_churn = data.groupby("favorite_category")["churn"].mean().sort_values() * 100
        fig3 = px.bar(
            cat_churn, orientation="h",
            title="Churn rate by favorite category (%)",
            labels={"value": "Churn rate (%)", "favorite_category": "Category"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Preview customer-level dataset"):
        st.dataframe(data.head(50), use_container_width=True)

# ------------------------------------------------------------------
# PAGE 2: PREDICT CHURN
# ------------------------------------------------------------------
elif page == "🔮 Predict Churn":
    st.title("🔮 Will this customer churn?")
    st.write("Fill in a customer's profile below and get an instant churn prediction.")

    with st.form("churn_form"):
        c1, c2 = st.columns(2)
        with c1:
            frequency = st.slider(FRIENDLY_LABELS["frequency"], 1, 10, 2)
            tenure_days = st.slider(FRIENDLY_LABELS["tenure_days"], 0, 900, 150)
            avg_order_value = st.number_input(
                FRIENDLY_LABELS["avg_order_value"], 0.0, 20000.0, 800.0, step=50.0)
            avg_discount_percent = st.slider(
                FRIENDLY_LABELS["avg_discount_percent"], 0, 60, 15)
            avg_rating = st.slider(FRIENDLY_LABELS["avg_rating"], 1.0, 5.0, 3.5, 0.1)
            avg_quantity = st.slider(FRIENDLY_LABELS["avg_quantity"], 1, 5, 3)
        with c2:
            return_rate = st.slider(FRIENDLY_LABELS["return_rate"], 0.0, 1.0, 0.1, 0.05)
            avg_shipping_cost = st.number_input(
                FRIENDLY_LABELS["avg_shipping_cost"], 0.0, 100.0, 15.0, step=1.0)
            favorite_category = st.selectbox(
                FRIENDLY_LABELS["favorite_category"], CATEGORICAL_OPTIONS["favorite_category"])
            customer_country = st.selectbox(
                FRIENDLY_LABELS["customer_country"], CATEGORICAL_OPTIONS["customer_country"])
            favorite_traffic_source = st.selectbox(
                FRIENDLY_LABELS["favorite_traffic_source"], CATEGORICAL_OPTIONS["favorite_traffic_source"])
            favorite_payment_method = st.selectbox(
                FRIENDLY_LABELS["favorite_payment_method"], CATEGORICAL_OPTIONS["favorite_payment_method"])

        submitted = st.form_submit_button("Predict churn", use_container_width=True)

    if submitted:
        raw_input = {
            "frequency": frequency,
            "tenure_days": tenure_days,
            "avg_order_value": avg_order_value,
            "avg_discount_percent": avg_discount_percent,
            "avg_rating": avg_rating,
            "return_rate": return_rate,
            "avg_shipping_cost": avg_shipping_cost,
            "avg_quantity": avg_quantity,
            "favorite_category": favorite_category,
            "customer_country": customer_country,
            "favorite_traffic_source": favorite_traffic_source,
            "favorite_payment_method": favorite_payment_method,
        }
        encoded_input = encode_row(raw_input)
        X = pd.DataFrame([encoded_input])[artifacts["clf_features"]]

        pred = artifacts["clf"].predict(X)[0]
        proba = artifacts["clf"].predict_proba(X)[0][1]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **Likely to churn** — churn probability: {proba*100:.1f}%")
        else:
            st.success(f"✅ **Likely to stay active** — churn probability: {proba*100:.1f}%")

        st.progress(min(int(proba * 100), 100))

        importances = pd.Series(
            artifacts["clf"].feature_importances_, index=artifacts["clf_features"]
        ).sort_values(ascending=True)
        importances.index = [FRIENDLY_LABELS.get(i, i) for i in importances.index]
        fig = px.bar(
            importances, orientation="h",
            title="What drives this model's churn predictions overall",
            labels={"value": "Importance", "index": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# PAGE 3: PREDICT CUSTOMER VALUE
# ------------------------------------------------------------------
elif page == "💰 Predict Customer Value":
    st.title("💰 How much revenue will this customer generate?")
    st.write(
        "Estimate a customer's total lifetime revenue based on their "
        "behavior so far."
    )

    with st.form("clv_form"):
        c1, c2 = st.columns(2)
        with c1:
            recency_days = st.slider(FRIENDLY_LABELS["recency_days"], 0, 900, 100)
            frequency = st.slider(FRIENDLY_LABELS["frequency"], 1, 10, 2, key="clv_freq")
            tenure_days = st.slider(FRIENDLY_LABELS["tenure_days"], 0, 900, 150, key="clv_tenure")
            avg_discount_percent = st.slider(
                FRIENDLY_LABELS["avg_discount_percent"], 0, 60, 15, key="clv_disc")
            avg_rating = st.slider(FRIENDLY_LABELS["avg_rating"], 1.0, 5.0, 3.5, 0.1, key="clv_rating")
            avg_quantity = st.slider(FRIENDLY_LABELS["avg_quantity"], 1, 5, 3, key="clv_qty")
        with c2:
            return_rate = st.slider(FRIENDLY_LABELS["return_rate"], 0.0, 1.0, 0.1, 0.05, key="clv_ret")
            avg_shipping_cost = st.number_input(
                FRIENDLY_LABELS["avg_shipping_cost"], 0.0, 100.0, 15.0, step=1.0, key="clv_ship")
            favorite_category = st.selectbox(
                FRIENDLY_LABELS["favorite_category"], CATEGORICAL_OPTIONS["favorite_category"], key="clv_cat")
            customer_country = st.selectbox(
                FRIENDLY_LABELS["customer_country"], CATEGORICAL_OPTIONS["customer_country"], key="clv_country")
            favorite_traffic_source = st.selectbox(
                FRIENDLY_LABELS["favorite_traffic_source"], CATEGORICAL_OPTIONS["favorite_traffic_source"], key="clv_traffic")
            favorite_payment_method = st.selectbox(
                FRIENDLY_LABELS["favorite_payment_method"], CATEGORICAL_OPTIONS["favorite_payment_method"], key="clv_pay")

        submitted = st.form_submit_button("Predict customer value", use_container_width=True)

    if submitted:
        raw_input = {
            "recency_days": recency_days,
            "frequency": frequency,
            "tenure_days": tenure_days,
            "avg_discount_percent": avg_discount_percent,
            "avg_rating": avg_rating,
            "return_rate": return_rate,
            "avg_shipping_cost": avg_shipping_cost,
            "avg_quantity": avg_quantity,
            "favorite_category": favorite_category,
            "customer_country": customer_country,
            "favorite_traffic_source": favorite_traffic_source,
            "favorite_payment_method": favorite_payment_method,
        }
        encoded_input = encode_row(raw_input)
        X = pd.DataFrame([encoded_input])[artifacts["reg_features"]]

        pred_revenue = artifacts["reg"].predict(X)[0]

        st.markdown("---")
        st.metric("Predicted total customer revenue (CLV)", f"${pred_revenue:,.2f}")

        pct = (data["total_revenue"] < pred_revenue).mean() * 100
        st.info(f"This places the customer in the **top {100-pct:.0f}%** of customers by predicted revenue.")

        fig = px.histogram(
            data, x="total_revenue", nbins=50,
            title="Where this prediction falls vs. all customers",
        )
        fig.add_vline(x=pred_revenue, line_color="red", line_width=3,
                       annotation_text="Prediction")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# PAGE 4: CUSTOMER SEGMENTS
# ------------------------------------------------------------------
elif page == "🧩 Customer Segments":
    st.title("🧩 Customer Segments (KMeans on RFM)")
    st.write(
        "Customers are grouped into segments based on **Recency**, "
        "**Frequency**, and **Monetary value** using K-Means clustering."
    )

    seg_summary = (
        data.groupby("segment_name")[["recency_days", "frequency", "total_revenue"]]
        .mean().round(1).reset_index()
        .rename(columns={
            "segment_name": "Segment", "recency_days": "Avg. recency (days)",
            "frequency": "Avg. orders", "total_revenue": "Avg. revenue ($)",
        })
    )
    seg_summary["Customers"] = data["segment_name"].value_counts().reindex(seg_summary["Segment"]).values
    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

    fig = px.scatter_3d(
        data.sample(min(3000, len(data)), random_state=42),
        x="recency_days", y="frequency", z="total_revenue",
        color="segment_name",
        title="Customer segments in RFM space",
        labels={"recency_days": "Recency (days)", "frequency": "Frequency",
                "total_revenue": "Total revenue ($)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔎 Find the segment for a new customer")
    c1, c2, c3 = st.columns(3)
    with c1:
        r = st.number_input("Days since last order", 0, 900, 100)
    with c2:
        f = st.number_input("Number of orders", 1, 20, 2)
    with c3:
        m = st.number_input("Total revenue so far ($)", 0.0, 50000.0, 1500.0, step=50.0)

    if st.button("Find segment", use_container_width=True):
        X_new = artifacts["cluster_scaler"].transform([[r, f, m]])
        seg_id = artifacts["kmeans"].predict(X_new)[0]
        seg_name = segment_name_map[seg_id]
        st.success(f"This customer belongs to the **{seg_name}** segment.")

    st.markdown("---")
    st.caption(
        "Segment names — Champions: recent, frequent, high spend · "
        "Loyal Customers: solid repeat buyers · At Risk: fading engagement · "
        "Lost / Dormant: long inactive."
    )
