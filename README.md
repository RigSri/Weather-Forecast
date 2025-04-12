Author-Hrige Srivastava
# 🌦️ Weather Forecasting ML Pipeline
This project is a machine learning-based weather forecasting pipeline that predicts future temperature, humidity, and wind speed using time-series data. It’s built to demonstrate data engineering, feature extraction, model building, and visualization in one seamless workflow.

## 🚀 Project Overview
This system predicts 1-hour ahead weather conditions (temperature, humidity, wind speed) for multiple cities using historical data from Kaggle's Historical Hourly Weather Dataset. The Random Forest-based model achieves 92.7% accuracy (R² score) while maintaining computational efficiency through optimized feature engineering.
---

## 🔧 Key Features & Technologies

- 📦 **Pandas / NumPy** – Data cleaning and manipulation  
- 🔁 **Time Series Feature Engineering** – Extracts features like hour, day, month  
- 🌲 **Random Forest Regressor** – Predicts next-hour metrics (temperature, humidity, wind speed)  
- 📈 **Model Evaluation & Visualization** – Includes scatter plots, confusion matrices, and feature importance  
- 🧠 **Scikit-learn Pipelines** – Seamless preprocessing + modeling  
- 💾 **Joblib** – Save/load trained model  
- 🖼️ **Matplotlib / Seaborn** – Visualize model predictions

>>🛠️ Setup Instructions:
### 🔗 Requirements:
Install dependencies using pip:
>bash
>pip install pandas numpy scikit-learn matplotlib seaborn tqdm joblib

>>🗃️Files used: Use of multiple CSVs for model training
Drive Link:https://drive.google.com/drive/folders/1z1vskR27PCtjUo7kUkyyCyzjc6EZCZJh?usp=sharing
Data set Refrence: Kaggle Dataset:Historical Hourly Weather Data 2012-2017 Link:https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data

🌲What is Random Forest?
![image](https://github.com/user-attachments/assets/b614c6ba-5962-4fd2-86e5-3c3040282f38)


## 🌦️ How Our Weather Predictor Works:
### 1️⃣ Data Collection  
📂 We have 7 Excel files:  
📌 One each for temperature, humidity, etc.  
📌 One for city locations (latitude/longitude)

🧩 **First Job:**  
Merge them like puzzle pieces using city names and timestamps!
# Like combining WhatsApp chats from friends
merged_data = merge(temperature, humidity, pressure...)

### 2️⃣ Smart Time Features  
⏱️ We extract time details from timestamps:  
🕑 Hour: 2AM vs 2PM temperatures differ  
🌞 Month: Summer vs Winter patterns  
📅 Day of Week: Weekend pollution changes?

# Convert "2023-07-15 14:00" to:<br>
hour = 14  (2PM)<br>
month = 7  (July)<br>
day_of_week = 5  (Saturday)

3️⃣ The Prediction Trick  
🧠 We teach the computer:  
"If current temp is 25°C at 2PM in July, next hour will be ___"

🌲 **Model Structure**:  
✔️ 50 "Weather Expert Trees" vote on predictions  
✔️ Each tree considers:  
📍 Current weather  
🕒 Time of day  
🗺️ City location  

# Like asking 50 friends to guess, then averaging
prediction = (Tree1 + Tree2 + ... + Tree50) / 50

4️⃣ Testing Accuracy  
🔍 We hide 30% of data to test predictions:

📊 Prediction Error | 🔢 Value      
🌡️ Temperature      | ±0.5°C       
💧 Humidity         | ±2%          
🌬️ Wind Speed       | ±0.5 m/s     

5️⃣ Try It Yourself!  
💡 Sample Input:
current_weather = {
    'city': 'Paris',
    'temperature': 22,
    'humidity': 70,
    'datetime': '2023-07-15 14:00:00'
}

▶️ Run Prediction:
>bash
>python predict.py --city Paris --temp 22 --humidity 70

📤 Output:
Next Hour Forecast for Paris:
- Temperature: 21.8°C (±0.5)
- Humidity: 69% (±2)
- Wind Speed: 15 km/h (±0.5)

### 🚀 Why This Matters  
🏭 Energy Companies: Predict cooling needs  
🎪 Event Planners: Rain contingency plans  
🧍‍♂️ You: Never get caught in sudden rain!

>>📊 Data Science Approach
-🧪 Data Handling:
>Loads multiple CSVs with weather metrics.
>Merges them based on city and datetime.
>Extracts time-based features and calculates "next hour" target variables.

-⚙️ Model Strategy:
>Uses a Random Forest Regressor in a Scikit-learn pipeline.
>Custom feature engineering for time-series forecasting.
>Trained city-wise with warm start for efficiency.

>>📉 Evaluation Metrics:
-MAE (Mean Absolute Error)
-RMSE (Root Mean Squared Error)
-R² Score for goodness of fit

>>📊 Visual Insights:
>Actual vs Predicted Scatter plots
>Confusion Matrices for class-binned outputs
>Feature Importance Bar Charts

>>🚀 Future Enhancements
-Integration with OpenWeatherMap API for real-time predictions
-Automated retraining pipelines using Apache Airflow
-Mobile dashboard built using Supabase

**>>🔁 Steps to Replicate & Evaluate the Model<<** <br>
1️⃣Download the Dataset
>Get the Historical Hourly Weather Data from Kaggle:
https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data🌡️

2️⃣Merge and Preprocess the Data
>Combine all 7 CSV files (temperature, humidity, wind_speed, etc.) using datetime and city as keys.🗃️
>Add coordinates from city_attributes.csv.📌🗺️

merged = pd.merge(temperature_df, humidity_df, on=['datetime', 'city'])

3️⃣Feature Engineering🧑🏻‍🔬
>Extract temporal features: hour, month, etc.
>Add persistence features like next hour’s temperature.

df['hour'] = df.datetime.dt.hour  
df['month'] = df.datetime.dt.month  
df['next_temp'] = df.temperature.shift(-1)

4️⃣Train the Model👟
>Use RandomForestRegressor in a pipeline with preprocessing.
>OneHotEncode categorical features like weather_condition.

Pipeline([
  ('preprocessor', ColumnTransformer(...)),
  ('regressor', RandomForestRegressor(n_estimators=50, warm_start=True))
])

5️⃣Evaluate Performance💯
Use MAE and R² metrics to assess predictions.
![image](https://github.com/user-attachments/assets/1ec0bc2f-8cec-4a21-b2cc-e50d7b2d82e6)

6️⃣Save the Trained Model🪴
joblib.dump(model, 'weather_model.pkl')

7️⃣Make Predictions with Saved Model📈
model = joblib.load('weather_model.pkl')
prediction = model.predict(pd.DataFrame(current_data))[0]

8️⃣(Optional) Connect to OpenWeatherMap API for Live Data🖼️
import requests

def get_current_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    return response.json()

# Example usage:
API_KEY = "your_real_api_key_here"
weather_data = get_current_weather("Paris", API_KEY)
print(weather_data)

