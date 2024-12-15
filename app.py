import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Debugging: Show current working directory
st.write(f"Current working directory: {os.getcwd()}")

# Page configuration
st.set_page_config(page_title="RainVision CBE 🌧️")

# Tabs for multiple sections
tab1, tab2 = st.tabs(["🌦️ RainVision CBE 🌧️", "📖 About Me"])

# Tab 1: RainVision CBE 🌧️
with tab1:
    st.title("☔ RainVision CBE App 🌦️")
    st.markdown(
        """
        Welcome to the **RainVision CBE App**! 🌧️  
        Select a region in Coimbatore and predict future rainfall based on historical data using **Machine Learning**. 📈
        """
    )
    
    # Dropdown with emojis for each area
    dat = st.selectbox(
        "📍 **Choose area**:",
        options=[
            '🌆 Gandhipuram', '🏠 R.S. Puram', '🏢 Townhall', '🏡 Saibaba Colony',
            '🏗️ Saravanampatti', '🏭 Ganapathy', '🚉 Podanur', '🌄 Kuniyamuthur',
            '🌳 Madukkarai', '🛤️ Singanallur', '🎓 Peelamedu', '🌾 Sulur',
            '🌴 Vadavalli', '⛰️ Thondamuthur', '⛪ Perur'
        ]
    )
    
    # Strip emoji to match file names
    dat = dat.split(' ', 1)[1]
    
    # Define file path based on selection
    file_path = f"Sample Data/{dat.replace(' ', '_').lower()}_data.csv"
    st.write(f"Checking for file: {file_path}")  # Debugging message
    
    # Check current working directory
    st.write(f"Current working directory: {os.getcwd()}")
    
    # Check absolute path
    absolute_path = os.path.abspath(file_path)
    st.write(f"Absolute path: {absolute_path}")
    
    # Check if the file exists
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            st.success(f"You selected **{dat}**. Data loaded successfully! ✅")
        except Exception as e:
            st.error(f"❌ Error loading the file: {e}")
    else:
        st.error(f"🚫 Data file not found for the selected area. Expected file: {file_path}.")
        
        # Allow file upload
        uploaded_file = st.file_uploader(f"🔄 **Upload the data file** for {dat}", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded and loaded successfully! ✅")
                st.write(df.head())  # Display first few rows to confirm
            except Exception as e:
                st.error(f"❌ Error loading the uploaded file: {e}")
                st.stop()
        else:
            st.stop()

    # Continue with the rest of the app after loading the data
    X_train = df.iloc[:, 0:1].values
    y_train = df.iloc[:, 1].values
    
    # Slider for year input
    data = st.slider("📅 **Select Year to Predict Rainfall:**", 2018, 2030, 2018)
    
    # Preparing test data
    X_test = np.array(data).reshape(-1, 1)
    
    # Train the model
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    
    # Predict the rainfall
    y_pred = regressor.predict(X_test)
    
    # Visualization
    fig, ax = plt.subplots()
    ax.plot(X_train, y_train, color='blue', label='Historical Data')
    ax.plot(X_test, y_pred, color='red', marker='o', label='Prediction')
    ax.set_title(f"RainVision CBE 🌧️ for {dat} 📊")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    st.pyplot(fig)
    
    # Display prediction result with formatting
    st.markdown(
        f"""
        ### 🌟 **Prediction Results**  
        For the year **{data}**, the predicted rainfall is:  
        ### 🌧️ `{y_pred[0]:.2f} mm` 🌧️
        """
    )

# Tab 2: About Me
with tab2:
    st.header("About Me")
    st.markdown(
        """
        # Hello! I'm **Niranjan NN** 👋  
        A passionate developer exploring the intersection of **AI** 🤖✨, **Data Science** 📊💡, and building tools that make a difference 🌟.
        """
    )
