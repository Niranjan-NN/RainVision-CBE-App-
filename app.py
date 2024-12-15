import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os  # To check file existence

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
    st.write(f"Trying to load file from: {file_path}")  # Debugging message
    
    # Read the selected area's data
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.success(f"You selected **{dat}**. Data loaded successfully! ✅")
    else:
        st.error(f"🚫 Data file not found for the selected area. Expected file: {file_path}.")
        uploaded_file = st.file_uploader("🔄 **Upload the data file**", type="csv")
        if uploaded_file is not None:
            # Try reading the uploaded file
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded and loaded successfully! ✅")
            except Exception as e:
                st.error(f"❌ Error loading the uploaded file: {e}")
                st.stop()
        else:
            st.stop()

    # Splitting the data for training
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
    
    # Additional expanders for insights
    with st.expander("📘 **Model Insights**"):
        st.markdown(
            """
            - **Model Used:** Random Forest Regressor 🌳  
            - **Historical Data:** Visualized in blue 📘  
            - **Predicted Value:** Represented in red 📕  
            """
        )
    
    with st.expander("📊 **Dataset Summary**"):
        st.dataframe(df.describe())

# Tab 2: About Me
with tab2:
    st.header("About Me")
    st.markdown(
        """
        # Hello! I'm **Niranjan NN** 👋  
        A passionate developer exploring the intersection of **AI** 🤖✨, **Data Science** 📊💡, and building tools that make a difference 🌟.
        
        ### About Me:
        - 🎓 **Pursuing Bachelor of Engineering** in Information Technology at **SNS College of Engineering**, Coimbatore, India.  
        - 💼 **Completed Internships** in **Data Science** and **Full-Stack Web Development** at **Codetech Solutions** and **Codsoft**.
        
        ### Why I Created RainVision CBE:
        The **RainVision CBE 🌦️** is a web-based tool that uses **machine learning** 🤖 to forecast annual rainfall 🌦️ in 15 areas of **Coimbatore**. It utilizes the **Random Forest Regressor** 🌳 to predict rainfall for the years 2018–2030 📅, offering a user-friendly interface 🖥️ with dynamic selection and visual representations 📊 of historical and predicted data. The app aims to support decision-making in **agriculture** 🌱, **urban planning** 🏙️, and **environmental management** 🌍.

        ### My Projects:
        1. 📝 **Smart Ration** – A time slot booking app for ration shops.  
        2. 🌏 **FieastaIndiana** – A tourism platform for hotel and guide bookings.  
        3. 🤖 **Aara** – An AI-powered chatbot for image recognition and insights.
        4. 🌿🕊️**InnerPeace AI**  - Your Anxiety Counselor 🎧.
        
        ### Interests:
        - 🌿 AI for Social Good  
        - 🛠️ Building impactful projects  
        - 📊 Data Science and Visualization  
        
        ### Connect With Me:
        - 🌐 **GitHub**: [github.com/niranjan](https://github.com/Niranjan-NN)  
        - 💼 **LinkedIn**: [linkedin.com/in/niranjan_nn](https://www.linkedin.com/in/niranjan-nn/)  
        """
    )
