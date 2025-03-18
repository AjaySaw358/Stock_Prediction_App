# 📈 Stock Prediction App

This is a Flask-based web application for predicting stock prices using a trained machine learning model.

## 🚀 Setup Instructions

Follow these steps to set up and run the project locally.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Stock_Prediction_App.git
cd Stock_Prediction_App
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

###  4️⃣ Run the Flask App
```bash
python app.py
```
🌍 Open the app in your browser:
👉 http://127.0.0.1:5000/
## 📌 How to Use
#### 1️⃣ Enter 40 stock feature values (10 steps × 4 indicators)
#### 2️⃣ Click "Predict"
#### 3️⃣ Get the next predicted stock price
#### 4️⃣ View the historical trend graph

**Example Input (40 values in Web UI)(yaml):**
```yaml
4300, 4400, 50, 1.2, 4310, 4410, 52, 1.3, 4320, 4420, 55, 1.4, 
4330, 4430, 58, 1.5, 4340, 4440, 60, 1.6, 4350, 4450, 63, 1.7, 
4360, 4460, 65, 1.8, 4370, 4470, 68, 1.9, 4380, 4480, 70, 2.0, 
4390, 4490, 72, 2.1
```

**API Usage**
POST Request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [[4300, 4400, 50, 1.2], [4310, 4410, 52, 1.3], ..., [4390, 4490, 72, 2.1]]}'
```
Response:
```bash
{
  "prediction": [[4395.6]]
}
```
