import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
import matplotlib.dates as mdates
import base64
import requests
import google.generativeai as genai
import os

# Define function to upload file to GitHub, creating a directory for each team
def upload_to_github(token, repo, team_name, path, content):
    """Uploads a file to the specified GitHub repository in a team-specific directory."""
    full_path = f"{team_name}/{path}"
    url = f"https://api.github.com/repos/{repo}/contents/{full_path}"
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    content_encoded = base64.b64encode(json.dumps(content).encode()).decode()
    data = {"message": f"Add {full_path}", "content": content_encoded, "branch": "main"}
    response = requests.put(url, headers=headers, json=data)
    if response.status_code == 201:
        st.success(f"File '{path}' uploaded to GitHub successfully in folder '{team_name}'.")
    else:
        st.error(f"Failed to upload to GitHub: {response.json().get('message', 'Unknown error')}")

# Load model parameters
@st.cache_resource
def load_model_parameters():
    model_path = '.streamlit/global_prophet_model_best.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("Model parameters loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model parameters: {e}")
        return None

# Initialize Gemini model once at the start of the app
gemini_api_key = st.secrets.get("GEMINI_API_KEY")
if gemini_api_key and "model" not in st.session_state:
    try:
        genai.configure(api_key=gemini_api_key)
        st.session_state.model = genai.GenerativeModel("gemini-1.5-flash")  # Store model in session state
        st.success("Ultra AI API Key successfully configured.")
    except Exception as e:
        st.error(f"An error occurred while setting up the Gemini model: {e}")

# Sidebar with Image, Navigation, and Instructions
st.sidebar.title("AI-Advisor")
page = st.sidebar.radio("Select Step:", ("1️⃣ Team and DB Connection", "2️⃣ Prediction & Buy Decision", 
                                         "3️⃣ AI Inventory Advisor", "4️⃣ Final Review & Feedback"))
st.sidebar.image(".streamlit/ai_image2.png", use_column_width=True)

# Maintain the current page in session state
if "current_page" not in st.session_state:
    st.session_state.current_page = page
elif page != st.session_state.current_page:
    st.session_state.current_page = page

# Define holidays DataFrame
holidays = pd.DataFrame({
    'holiday': 'sales_event',
    'ds': pd.to_datetime([
        '2024-01-05', '2024-01-07', '2024-01-09', '2024-01-10', '2024-01-11',
        '2024-01-15', '2024-01-17', '2024-01-22', '2024-01-25', '2024-01-26',
        '2024-01-27', '2024-01-31', '2024-02-05', '2024-02-07', '2024-02-09',
        '2024-02-10', '2024-02-11', '2024-02-15', '2024-02-17'
    ]),
    'lower_window': 0,
    'upper_window': 1,
})

# Feature engineering
def engineer_features(df):
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['day_of_year'] = df['ds'].dt.dayofyear
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
    return df

# Smoothing function
def smooth_predictions(predictions, window_size=11):
    return predictions.rolling(window=window_size, min_periods=1).mean()

# Generate predictions with performance adjustments based on trust responses
def generate_predictions(game_data, forecast_start_day, forecast_end_day, trust_responses):
    model = Prophet(holidays=holidays)
    game_data['ds'] = pd.to_datetime(game_data['ds'], errors='coerce')
    
    # Replace NaN or missing values with zero
    game_data['y'].fillna(0, inplace=True)

    # Fit model
    try:
        model.fit(game_data)
    except Exception as e:
        st.error(f"Error fitting the model: {e}")
        return None

    # Define forecast period
    future_dates = pd.date_range(start=game_data['ds'].max() + pd.Timedelta(days=1), periods=forecast_end_day - forecast_start_day + 1)
    future = pd.DataFrame({'ds': future_dates})
    future = engineer_features(future)
    
    # Predict sales
    forecast = model.predict(future)
    
    # Ensure predictions are positive
    forecast['yhat'] = forecast['yhat'].clip(lower=1)

    # Apply smoothing to the predictions
    forecast['yhat'] = smooth_predictions(forecast['yhat'])

    # Adjust predictions based on trust responses
    forecast['adjusted_yhat'] = forecast['yhat']
    for i, response in enumerate(trust_responses):
        if i < len(forecast):
            if response == 'AA':
                forecast.loc[i, 'adjusted_yhat'] = forecast.loc[i, 'yhat']
            elif response == 'BB':
                noise_factor = np.random.uniform(0.7, 1.0)
                forecast.loc[i, 'adjusted_yhat'] = forecast.loc[i, 'yhat'] * noise_factor

    forecast['adjusted_yhat'] = forecast['adjusted_yhat'].clip(lower=1)
    return forecast[['ds', 'yhat', 'adjusted_yhat']]

# Retrieve all sales data with complete game days from COM_day table
def get_game_data(host, port, database, user, password):
    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        st.success("Database connection successful.")
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None, None

    table_name = f"LN{user.split('_')[-1]}_sales"
    try:
        # Query to get all game dates from COM_day
        com_day_query = "SELECT date FROM COM_day ORDER BY date DESC;"
        com_day_df = pd.read_sql_query(com_day_query, conn)

        # Query to get all sales data from sales table
        sales_query = f"SELECT date, unit_sold FROM {table_name} ORDER BY date DESC;"
        sales_df = pd.read_sql_query(sales_query, conn)

        # Full join sales data with COM_day dates on 'date' column
        game_data = pd.merge(com_day_df, sales_df, on="date", how="left").sort_values('date')
        
        # Fill any missing 'unit_sold' values with 0
        game_data['unit_sold'].fillna(0, inplace=True)

        st.success("Data retrieval successful.")
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
        return None, None
    finally:
        conn.close()

    # Convert date to datetime and handle invalid dates
    game_data['ds'] = pd.to_datetime(game_data['date'], errors='coerce')
    
    # Drop rows with NaN in 'ds' (which couldn't be converted to dates)
    game_data = game_data.dropna(subset=['ds']).sort_values('ds')

    # Rename columns for Prophet
    game_data = game_data.rename(columns={'unit_sold': 'y'}).sort_values('ds')
    current_game_day = (game_data['ds'].max() - pd.to_datetime("2024-01-01")).days + 1
    return game_data, current_game_day



# Step 1: Team and Database Connection
if st.session_state.current_page == "1️⃣ Team and DB Connection":
    st.title("Connect to Database")
    st.markdown("#### Step 1: Enter Database and Team Information")
    team_name = st.text_input("Team Name", help="Enter your team name for identification.")
    user = st.text_input("User", help="Your unique user ID (e.g., LN69130_342409).")
    host = st.text_input("Host", help="Database host URL.")
    port = st.text_input("Port", value="3306", help="Port number, typically 3306 for MySQL.")
    database = st.text_input("Database Name", help="The name of your MySQL database.")
    password = st.text_input("Password", help="Your MySQL password.")

    if st.button("Save & Continue"):
        st.session_state.update({"team_name": team_name, "user": user, "host": host, "port": port, 
                                 "database": database, "password": password})
        st.success("Settings saved successfully! Moving to the next step.")
        st.session_state.current_page = "2️⃣ Prediction & Buy Decision"

# Step 2: Prediction and Buy Decision
elif st.session_state.current_page == "2️⃣ Prediction & Buy Decision":
    st.title("Predict Sales and Make Buy Decisions")
    st.markdown("#### Step 2: Review Predictions and Decide on Purchases")

    current_stock = st.number_input("Current Stock", min_value=0, help="Enter the current stock available in units.")
    st.session_state.current_stock = current_stock

    trust_chain_responses = ['AA', 'AA', 'BB', 'BB', 'AA', 'BB', 'AA']
    if 'historical_forecasts' not in st.session_state:
        st.session_state.historical_forecasts = []

    if st.button("Show Prediction"):
        game_data, current_game_day = get_game_data(
            st.session_state.host,
            st.session_state.port,
            st.session_state.database,
            st.session_state.user,
            st.session_state.password
        )

        forecast = None
        if current_game_day:
            forecast_ranges = {
                14: (15, 19),
                19: (20, 24),
                24: (25, 29),
                29: (30, 34),
                34: (35, 39),
                39: (40, 44),
                44: (45, 49),
            }
            for day, (start, end) in forecast_ranges.items():
                if current_game_day == day:
                    forecast = generate_predictions(game_data, start, end, trust_chain_responses)
                    st.session_state.historical_forecasts.append(forecast)
                    st.session_state.forecast_start_day = start
                    st.session_state.forecast_end_day = end

        if forecast is not None:
            total_predicted_sales = forecast['adjusted_yhat'].sum()
            st.session_state.total_predicted_sales = total_predicted_sales

            plt.figure(figsize=(20, 16))
            game_data['y'] = pd.to_numeric(game_data['y'], errors='coerce')
            game_data_clean = game_data.dropna(subset=['y'])

            if not game_data_clean.empty:
                plt.plot(game_data_clean['ds'], game_data_clean['y'], label="Actual Sales", marker='o', color='black', linewidth=2)
                for x, y in zip(game_data_clean['ds'], game_data_clean['y']):
                    plt.text(x, y, f'{y:.0f}', ha='center', va='bottom', fontsize=16, color='black')

            plotted_forecasts = set()
            colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown']
            
            for idx, hist_forecast in enumerate(st.session_state.historical_forecasts[:-1]):
                if not hist_forecast.empty:
                    color = colors[idx % len(colors)]
                    forecast_label = f"Forecast (Days {hist_forecast['ds'].min().day}-{hist_forecast['ds'].max().day})"
                    if forecast_label not in plotted_forecasts:
                        plt.plot(hist_forecast['ds'], hist_forecast['adjusted_yhat'], label=forecast_label, linestyle='--', marker='x', color=color, linewidth=1.5)
                        plotted_forecasts.add(forecast_label)
                        for x, y in zip(hist_forecast['ds'], hist_forecast['adjusted_yhat']):
                            plt.text(x, y, f'{y:.0f}', ha='center', va='top', fontsize=16, color=color)

            forecast_clean = forecast.dropna(subset=['adjusted_yhat'])
            if not forecast_clean.empty:
                plt.plot(forecast_clean['ds'], forecast_clean['adjusted_yhat'], label="Current Forecast", linestyle='-', marker='s', color='tab:blue', linewidth=2)
                for x, y in zip(forecast_clean['ds'], forecast_clean['adjusted_yhat']):
                    plt.text(x, y, f'{y:.0f}', ha='center', va='top', fontsize=16, color='tab:blue')

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.gcf().autofmt_xdate()
            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
            plt.xlabel('Day', fontsize=12)
            plt.ylabel('Units Sold', fontsize=12)
            plt.title("Actual vs Predicted Sales Over Time", fontsize=12)
            
            plt.legend(title="Sales Data", loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 14}, title_fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            st.pyplot(plt.gcf())
            st.write(f"**Total Predicted Sales for Days {forecast_clean['ds'].min().day} to {forecast_clean['ds'].max().day}: {total_predicted_sales:.0f} units**")

    units_to_buy = st.number_input("Units to Buy", min_value=0, step=1, help="Enter the quantity of units you plan to purchase.")
    if st.button("Save Purchase Decision"):
        st.session_state.units_to_buy = units_to_buy
        st.success("Purchase decision saved!")

# Step 3: Discount and Purchase Threshold
elif st.session_state.current_page == "3️⃣ AI Inventory Advisor":
    st.title("Get Advice from AI")
    st.markdown("#### Step 3: Get Advice from AI")
    
    current_stock = st.session_state.get("current_stock", 0)

    discount_percentage = st.text_input("Discount Percentage", help="Enter discount percentage as a whole number.")
    min_purchase_quantity = st.number_input("Minimum Purchase Quantity", min_value=0, help="Minimum purchase quantity to proceed.")

    if st.button("Get Advice from AI"):
        if "model" in st.session_state:
            prompt = (
                "You are an inventory management expert. Based on the forecasted sales data provided below, "
                "calculate an optimal restock quantity to ensure stock availability over the next 5 days, while factoring in "
                "the current stock level:\n\n"
                
                f"**Total Predicted Sales for Days {st.session_state.forecast_start_day} to {st.session_state.forecast_end_day}:** "
                f"{st.session_state.total_predicted_sales:.0f} units (Note: forecasted demand may vary slightly from actual demand, "
                "so consider a margin of error in your calculations.)\n"
                f"**Current Stock:** {current_stock} units\n"
                f"**Player's Planned Units to Buy:** {st.session_state.units_to_buy} units\n"
                f"**Minimum Purchase Quantity (Discount Threshold):** {min_purchase_quantity} units\n"
                f"**Discount Percentage (Low Priority):** {discount_percentage}%\n\n"
                
                "Consider these parameters:\n"
                "• **Vendor Order Quantities Available:** 1,000; 3,000; 5,000; 8,000; 12,000; 15,000; 20,000; 30,000; 40,000.\n"
                "  Note: these represent single-purchase options that can be repeated multiple times to reach the required stock level.\n"
                "• **Lead Time:** 1 day\n"
                "• **Shelf Life:** 8 days.\n\n"
                
                "Provide a recommendation that prioritizes maintaining stock levels to meet sales demand, while taking into account "
                "the current stock level in the calculation. Avoid matching the player’s planned {st.session_state.units_to_buy} exactly; "
                "adjust as necessary to cover any variability in the forecasted demand. Additionally, explain why adjusting the order size "
                "to your recommendation may better meet sales demand, reduce stockouts, or improve inventory turnover.\n\n"
                
                "Present your recommendation in the following table format:\n\n"
                
                "| Parameter                         | Value                                     |\n"
                "|-----------------------------------|-------------------------------------------|\n"
                "| Recommended Order Size            | <Your Recommendation Here>               |\n"
                "| Rationale                         | <Reasoning Here>                         |\n"
                "| Comparison with Player's Plan     | <Comparison Here>                        |\n"
                "| Estimated Stock Turnover Days     | <Estimation Here>                        |\n"
                "| Forecasted Demand Coverage Days   | <Coverage Here>                          |\n"
                "| Comments                          | <Additional Notes>                       |\n\n"
                
                "Please fill in the table with a clear, actionable recommendation for the restock quantity that prioritizes "
                "meeting the forecasted demand and outlines the benefits of adjusting the order size as needed. Emphasize "
                "the impact of current stock, demand variability, and the option to purchase multiple vendor order quantities in your reasoning. Explanation is not necessary."
            )


            try:
                response = st.session_state.model.generate_content(prompt)
                st.session_state.AI_advice = response.text
                st.write(f"AI Advice:\n\n{response.text}")
                st.success("Advice generated successfully.")
                st.session_state.advice_generated = True
            except Exception as e:
                st.error(f"An error occurred while generating advice: {e}")
        else:
            st.error("Model not initialized. Please check API key configuration.")

# Step 4: Final Review and Feedback
elif st.session_state.current_page == "4️⃣ Final Review & Feedback":
    st.title("Review Final Purchase and Provide Feedback")
    st.markdown("#### Step 4: Confirm Purchase and Answer Survey")
    
    final_quantity = st.number_input("Final Purchase Quantity", min_value=0, help="Final quantity to purchase based on AI advice and analysis.")
    st.write("### Feedback on AI Assistance")
    feedback_options = ["1 (Strongly Disagree)", "2", "3", "4", "5", "6", "7 (Strongly Agree)"]

    q1_response = st.selectbox("I relied on AI advice", options=feedback_options)
    q2_response = st.selectbox("I agree with AI advice", options=feedback_options)
    q3_response = st.selectbox("I trusted AI in the game tasks", options=feedback_options)

    if st.button("Save Results"):
        if "save_count" not in st.session_state:
            st.session_state.save_count = 0
        st.session_state.save_count += 1

        filename = f"{st.session_state.team_name}_final_result_{st.session_state.save_count}.json"
        result_data = {
            "final_purchase_quantity": final_quantity,
            "AI_advice": st.session_state.get("AI_advice", ""),
            "units_to_buy": st.session_state.get("units_to_buy", ""),
            "questions": {
                "I relied on AI advice": q1_response,
                "I agree with AI advice": q2_response,
                "I trusted AI in the game tasks": q3_response
            },
            "team_name": st.session_state.team_name,
            "user": st.session_state.user,
            "host": st.session_state.host,
            "port": st.session_state.port,
            "database": st.session_state.database,
            "password": st.session_state.password
        }

        with open(filename, 'w') as f:
            json.dump(result_data, f)

        github_token = st.secrets["GITHUB_TOKEN"]
        repo = "ginga924/monsoon-advisor"
        upload_to_github(github_token, repo, st.session_state.team_name, filename, result_data)
        
        st.success("Result saved successfully!")
        st.session_state.current_page = "2️⃣ Prediction & Buy Decision"  # Loop back to Step 2