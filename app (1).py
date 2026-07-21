import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Models
# -----------------------------

regressor = joblib.load("random_forest_regressor.pkl")
classifier = joblib.load("random_forest_classifier.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Title
# -----------------------------

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction")

menu = st.sidebar.selectbox(
    "Select Module",
    (
        "Home",
        "Random Forest Regressor",
        "Random Forest Classifier",
        "Customer Segmentation"
    )
)

# ====================================================
# HOME
# ====================================================

if menu == "Home":

    st.header("Welcome")

    st.write("""
This dashboard contains

- Random Forest Regressor
- Random Forest Classifier
- K-Means Customer Segmentation
""")

# ====================================================
# REGRESSOR
# ====================================================

elif menu == "Random Forest Regressor":

    st.header("Profit Prediction")

    product_category = st.number_input("Product Category (Encoded)", 0)

    product_price = st.number_input("Product Price", 0.0)

    discount_percent = st.slider("Discount %",0,100)

    quantity = st.number_input("Quantity",1)

    customer_country = st.number_input("Customer Country (Encoded)",0)

    traffic_source = st.number_input("Traffic Source (Encoded)",0)

    payment_method = st.number_input("Payment Method (Encoded)",0)

    shipping_cost = st.number_input("Shipping Cost",0.0)

    rating = st.slider("Rating",1.0,5.0)

    is_returned = st.selectbox("Returned",[0,1])

    discounted_price = st.number_input("Discounted Price",0.0)

    revenue = st.number_input("Revenue",0.0)

    if st.button("Predict Profit"):

        input_data = [[
            product_category,
            product_price,
            discount_percent,
            quantity,
            customer_country,
            traffic_source,
            payment_method,
            shipping_cost,
            rating,
            is_returned,
            discounted_price,
            revenue
        ]]

        prediction = regressor.predict(input_data)

        st.success(f"Predicted Profit : {prediction[0]:.2f}")

# ====================================================
# CLASSIFIER
# ====================================================

elif menu == "Random Forest Classifier":

    st.header("Return Prediction")

    product_category = st.number_input("Product Category",0,key=1)

    product_price = st.number_input("Product Price",0.0,key=2)

    discount_percent = st.slider("Discount",0,100,key=3)

    quantity = st.number_input("Quantity",1,key=4)

    customer_country = st.number_input("Customer Country",0,key=5)

    traffic_source = st.number_input("Traffic Source",0,key=6)

    payment_method = st.number_input("Payment Method",0,key=7)

    shipping_cost = st.number_input("Shipping Cost",0.0,key=8)

    rating = st.slider("Rating",1.0,5.0,key=9)

    discounted_price = st.number_input("Discounted Price",0.0,key=10)

    revenue = st.number_input("Revenue",0.0,key=11)

    profit = st.number_input("Profit",0.0,key=12)

    if st.button("Predict Return"):

        input_data = [[
            product_category,
            product_price,
            discount_percent,
            quantity,
            customer_country,
            traffic_source,
            payment_method,
            shipping_cost,
            rating,
            discounted_price,
            revenue,
            profit
        ]]

        prediction = classifier.predict(input_data)

        if prediction[0] == 1:
            st.error("Product likely to be Returned")
        else:
            st.success("Product likely NOT to be Returned")

# ====================================================
# KMEANS
# ====================================================

elif menu == "Customer Segmentation":

    st.header("Customer Segmentation")

    revenue = st.number_input("Revenue",0.0,key=20)

    profit = st.number_input("Profit",0.0,key=21)

    quantity = st.number_input("Quantity",1,key=22)

    rating = st.slider("Rating",1.0,5.0,key=23)

    discount = st.slider("Discount",0,100,key=24)

    shipping = st.number_input("Shipping Cost",0.0,key=25)

    if st.button("Find Cluster"):

        data = [[
            revenue,
            profit,
            quantity,
            rating,
            discount,
            shipping
        ]]

        data = scaler.transform(data)

        cluster = kmeans.predict(data)

        st.success(f"Customer belongs to Cluster {cluster[0]}")
