import streamlit as st
import pandas as pd

from predict import predict_survival


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🚢 Titanic ML Project")

st.sidebar.markdown("""
## Model Information

- **Algorithm:** Logistic Regression
- **Dataset:** Titanic Survival Dataset
- **Framework:** Streamlit
- **Language:** Python
- **Author:** Nayab Yaseen Khan
""")

st.sidebar.divider()

st.sidebar.metric(
    "Model Accuracy",
    "82%"
)


# -----------------------------
# Main Title
# -----------------------------
st.title("🚢 Titanic Survival Prediction AI")

st.write(
    "This application predicts whether a passenger would survive on the Titanic using a Machine Learning Logistic Regression Model."
)

st.divider()


# -----------------------------
# Passenger Details
# -----------------------------
st.subheader("👤 Passenger Details")


col1, col2 = st.columns(2)


with col1:

    pclass = st.selectbox(
        "🎫 Passenger Class",
        [1, 2, 3]
    )

    gender = st.selectbox(
        "👤 Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "🎂 Age",
        min_value=0,
        max_value=100,
        value=25
    )

    sibsp = st.number_input(
        "👨‍👩‍👧 Siblings / Spouse",
        min_value=0,
        value=0
    )


with col2:

    parch = st.number_input(
        "👨‍👩‍👦 Parents / Children",
        min_value=0,
        value=0
    )

    fare = st.number_input(
        "💰 Fare",
        min_value=0.0,
        value=50.0
    )

    embarked = st.selectbox(
        "🚢 Embarked",
        ["Southampton", "Cherbourg", "Queenstown"]
    )


st.divider()


# -----------------------------
# Predict Button
# -----------------------------
predict_button = st.button(
    "🔍 Predict Survival",
    use_container_width=True
)


# -----------------------------
# Prediction
# -----------------------------
if predict_button:

    # Gender Encoding
    sex = 0 if gender == "Male" else 1


    # Embarked Encoding
    embarked_map = {
        "Southampton": 0,
        "Cherbourg": 1,
        "Queenstown": 2
    }

    embarked_value = embarked_map[embarked]


    # Create Input Data
    input_data = pd.DataFrame({

        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked_value]

    })


    # Get Prediction From predict.py
    prediction, probability = predict_survival(input_data)


    st.divider()


    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.subheader("🤖 Prediction Result")


    if prediction == 1:

        st.success(
            "🎉 Passenger is predicted to SURVIVE."
        )

        confidence = probability[1] * 100


    else:

        st.error(
            "❌ Passenger is predicted NOT to SURVIVE."
        )

        confidence = probability[0] * 100



    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    st.divider()


    # -----------------------------
    # Passenger Summary
    # -----------------------------
    st.subheader("📋 Passenger Summary")


    st.markdown(
        f"""
        | Details | Value |
        |---------|-------|
        | 👤 Gender | {gender} |
        | 🎫 Passenger Class | {pclass} |
        | 🎂 Age | {age} |
        | 👨‍👩‍👧 Siblings / Spouse | {sibsp} |
        | 👨‍👩‍👦 Parents / Children | {parch} |
        | 💰 Fare | ₹{fare} |
        | 🚢 Embarked | {embarked} |
        """,
        unsafe_allow_html=True
    )