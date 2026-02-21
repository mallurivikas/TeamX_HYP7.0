# 🏥 Multimodal Health Assessment System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/mongodb-atlas-brightgreen.svg)](https://www.mongodb.com/cloud/atlas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An intelligent, AI-powered health assessment platform that combines machine learning, computer vision, and natural language processing to provide comprehensive health risk analysis and personalized wellness recommendations.**

---

## 🌟 What is This System?

Imagine having a personal health assistant that can:
- 📊 **Analyze your health metrics** (blood pressure, glucose, BMI, etc.)
- 🤖 **Predict disease risks** using AI (diabetes, heart disease, hypertension, obesity)
- 📸 **Read your face** to detect stress and pain levels
- 📄 **Extract data from medical reports** automatically
- 🥗 **Scan nutrition labels** and give instant health advice
- 💬 **Chat via WhatsApp** for quick health queries
- 📱 **Generate personalized health plans** with AI recommendations

This system does all of that! It's like having a smart doctor, nutritionist, and wellness coach in one application.

---

## 🎯 Who Is This For?

### 👥 **Target Users**
- **Health-conscious individuals** who want to monitor their wellness
- **Patients** managing chronic conditions (diabetes, hypertension, obesity)
- **Healthcare providers** looking for preliminary screening tools
- **Researchers** exploring multimodal health assessment
- **Developers** learning about ML, computer vision, and healthcare AI

### ⚕️ **Use Cases**
1. **Preventive Health Screening** - Catch risks before they become problems
2. **Chronic Disease Management** - Track and manage ongoing conditions
3. **Nutrition Guidance** - Make better food choices with AI-powered scanning
4. **Remote Health Monitoring** - WhatsApp integration for accessibility
5. **Educational Tool** - Learn about health risks and wellness

---

## ✨ Key Features Explained

### 1️⃣ **Multi-Disease Risk Prediction** 🎯

The system uses **4 specialized AI models** to assess your health:

| Model | What It Predicts | Key Factors Analyzed |
|-------|-----------------|---------------------|
| 🩸 **Diabetes** | Risk of developing diabetes | Glucose, insulin, BMI, age, family history |
| ❤️ **Heart Disease** | Cardiovascular health risks | Cholesterol, blood pressure, heart rate, chest pain |
| 🩺 **Hypertension** | High blood pressure risks | Systolic/diastolic BP, age, lifestyle, stress |
| ⚖️ **Obesity** | Weight-related health issues | BMI, eating habits, physical activity, calories |

**How it works:**
```
Your Input → AI Models → Individual Risk Scores → Combined Health Score
     ↓                                                        ↓
 (age, BP, glucose, etc.)                          (0-100 with letter grade)
```

**Example Output:**
```
Overall Health Score: 72/100 (Grade: C)
Risk Level: Moderate

Individual Risks:
├─ Heart Disease: 45% (Moderate)
├─ Diabetes: 38% (Low-Moderate)
├─ Hypertension: 52% (Moderate)
└─ Obesity: 35% (Low-Moderate)
```

---

### 2️⃣ **Facial Expression Analysis** 📸

Uses **computer vision** to analyze your face in real-time:

- **Pain Detection** - Identifies facial cues indicating discomfort
- **Stress Levels** - Analyzes micro-expressions for stress markers
- **Anxiety Assessment** - Detects signs of anxiety through facial features

**Technology Stack:**
- MediaPipe for facial landmark detection (468 points on face)
- Custom ML algorithms for expression classification
- Real-time webcam processing

**Privacy Note:** All processing happens on your device. No images are stored or uploaded.

---

### 3️⃣ **Medical Report Extraction** 📄

Upload a PDF medical report, and the AI automatically extracts:

- ✅ Blood test results (glucose, cholesterol, hemoglobin, etc.)
- ✅ Vital signs (blood pressure, heart rate, temperature)
- ✅ Patient information (age, gender)
- ✅ Medical history and conditions

**Supported Report Types:**
- Blood Test Reports
- Lipid Panels
- Diabetes Screening
- General Health Checkups

**How it works:**
1. Upload PDF → 2. OCR extracts text → 3. AI identifies values → 4. Auto-fills assessment form

---

### 4️⃣ **Nutrition Label Scanner** 🥗

Scan any food product's nutrition label and get instant insights:

**What You Get:**
- 🚦 **Health Score** (0-100)
- ⚠️ **Warnings** (high sugar, sodium, artificial ingredients)
- ✅ **Positives** (fiber, protein, vitamins)
- 💡 **Recommendations** (better alternatives, portion control)
- 🔢 **Detailed Breakdown** (calories, macros, micronutrients)

**Example:**
```
Product: Chocolate Chip Cookies
Health Score: 35/100 (Poor)

⚠️ Concerns:
- High sugar (45g per serving - exceeds daily limit)
- High saturated fat (12g)
- Low fiber (1g)

✅ Better Choice: Oatmeal cookies with less sugar
```

---

### 5️⃣ **WhatsApp Integration** 💬

Get health advice right in WhatsApp!

**Commands:**
- `Hello` - Start conversation
- Send a nutrition label photo - Get instant analysis
- Ask health questions - Get AI-powered answers

**Use Cases:**
- Quick nutrition checks while shopping
- Emergency health queries
- Remote monitoring for elderly/patients

---

### 6️⃣ **Personalized Health Plans** 📋

Based on your assessment, the AI generates:

- 🎯 **Custom Goals** (weight loss, BP control, glucose management)
- 🍽️ **Diet Recommendations** (meal plans, foods to eat/avoid)
- 🏃 **Exercise Plans** (type, duration, intensity)
- 💊 **Lifestyle Modifications** (sleep, stress management, habits)
- 📅 **Action Timeline** (daily, weekly, monthly goals)

**Powered by:** Google Gemini AI (1.5 Flash model)

---

## 🚀 Getting Started

### Prerequisites

Before you begin, make sure you have:

- **Python 3.8+** installed ([Download here](https://www.python.org/downloads/))
- **MongoDB Atlas account** (free tier works) ([Sign up](https://www.mongodb.com/cloud/atlas/register))
- **Google Gemini API key** ([Get it here](https://makersuite.google.com/app/apikey))
- **(Optional) Twilio account** for WhatsApp features ([Sign up](https://www.twilio.com/try-twilio))

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/multimodal-health-system.git
cd "multimodal system"
```

---

### Step 2: Install Dependencies

```bash
# Install all required Python packages
pip install -r requirements.txt
```

**What gets installed:**
- `flask` - Web framework
- `scikit-learn` - Machine learning models
- `opencv-python` - Computer vision
- `mediapipe` - Facial analysis
- `pymongo` - MongoDB database
- `google-generativeai` - Gemini AI
- `twilio` - WhatsApp integration
- Many more...

---

### Step 3: Configure Environment Variables

1. **Copy the example file:**
```bash
cp .env.example .env
```

2. **Edit `.env` and add your credentials:**

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/health_db

# Flask Configuration
FLASK_SECRET_KEY=your-super-secret-key-here-change-me

# Google Gemini API (for AI health plans)
GEMINI_API_KEY=your-gemini-api-key-here

# Twilio (for WhatsApp - Optional)
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
```

**Where to get these:**
- **MongoDB URI**: MongoDB Atlas → Clusters → Connect → Connect your application
- **Gemini API**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Twilio**: [Twilio Console](https://console.twilio.com/)

---

### Step 4: Train the AI Models

**First time only** - train the health prediction models:

```bash
python train_all_models.py
```

**What happens:**
- Loads health datasets (`diabetes.csv`, `heart.csv`, etc.)
- Trains 4 machine learning models
- Saves trained models to `saved_models/` folder
- Shows accuracy scores for each model

**Expected output:**
```
🏋️ Training Diabetes Model...
✅ Diabetes Model trained successfully! Accuracy: 76.32%

🏋️ Training Heart Disease Model...
✅ Heart Model trained successfully! Accuracy: 85.18%

🏋️ Training Hypertension Model...
✅ Hypertension Model trained successfully! Accuracy: 94.56%

🏋️ Training Obesity Model...
✅ Obesity Model trained successfully! Accuracy: 96.12%

✨ All models trained and saved successfully!
```

**Time required:** 1-2 minutes (only needed once)

---

### Step 5: Run the Application

```bash
python app.py
```

**Expected output:**
```
✅ Pipeline and Nutrition Analyzer initialized successfully!
✅ Multimodal agents initialized successfully!
⚠️ X-Ray Analyzer is disabled
🌐 Starting Health Assessment Web Application...
📱 Open your browser and navigate to: http://localhost:5000
⚕️  Remember: This is for educational purposes only!
 * Running on http://0.0.0.0:5000
```

---

### Step 6: Open in Browser

Visit: **http://localhost:5000**

You'll see the login page. Create an account to get started!

---

## 📖 User Guide

### 🔐 **Registration & Login**

1. **Create Account** → Enter username, email, password
2. **Login** → Use your credentials
3. **Profile** → View your assessment history

**Security Features:**
- Passwords are hashed (bcrypt)
- Session management
- Secure MongoDB storage

---

### 📊 **Taking a Health Assessment**

#### **Option A: Quick Assessment** (5 minutes)

1. Navigate to **Quick Check** from menu
2. Enter basic info:
   - Age, gender, height, weight
   - Blood pressure, glucose, cholesterol
3. Click **Analyze Health**
4. View instant results

#### **Option B: Full Assessment** (10-15 minutes)

1. Navigate to **Full Assessment**
2. Complete all sections:
   - **Personal Info** (age, gender, height, weight)
   - **Vital Signs** (BP, heart rate, temperature)
   - **Blood Tests** (glucose, cholesterol, insulin, etc.)
   - **Lifestyle** (smoking, alcohol, exercise, sleep)
   - **Diet Habits** (meals, water, vegetables, processed food)
   - **Medical History** (family history, conditions)
3. **(Optional)** Enable webcam for facial analysis
4. Submit assessment
5. View comprehensive results with:
   - Health score & grade
   - Individual disease risks
   - Risk level classification
   - Detailed recommendations
   - Downloadable PDF report

---

### 📄 **Uploading Medical Reports**

1. Click **Upload Report** from dashboard
2. Select report type (Blood Test, Lipid Panel, etc.)
3. Upload PDF file
4. AI extracts values automatically
5. Review and confirm extracted data
6. Proceeds to assessment with pre-filled values

**Supported Formats:** PDF (max 16MB)

---

### 🥗 **Scanning Nutrition Labels**

1. Navigate to **Nutrition Scanner**
2. Options:
   - **Upload Image** - From your device
   - **Take Photo** - Use webcam
3. AI analyzes label
4. View results:
   - Health score
   - Warnings & positives
   - Detailed nutrients
   - Recommendations
5. Export report as needed

---

### 💬 **Using WhatsApp Features**

1. **Setup** (Admin only):
   - Configure Twilio credentials in `.env`
   - Set webhook URL in Twilio console
   - See `WHATSAPP_SETUP.md` for details

2. **User Experience**:
   - Send "Hello" to bot number
   - Send nutrition label photo
   - Receive instant analysis
   - Ask health questions

---

## 🏗️ System Architecture

### **High-Level Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB INTERFACE (Flask)                   │
│  Login/Register │ Dashboard │ Assessment │ Results │ Profile │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┬──────────────┬─────────────┐
    │                         │              │             │
┌───▼────┐           ┌────────▼──────┐  ┌───▼────┐   ┌───▼────┐
│   ML   │           │  Multimodal   │  │ Gemini │   │WhatsApp│
│ Models │           │    Agents     │  │   AI   │   │  Bot   │
│        │           │               │  │        │   │        │
│ • Diabetes         │ • Facial      │  │ Health │   │ Twilio │
│ • Heart │           │ • Report OCR  │  │ Plans  │   │ API    │
│ • Hyper│           │ • Nutrition   │  └────────┘   └────────┘
│ • Obesity          │   Scanner     │
└────────┘           └───────────────┘
     │                       │
     └───────────┬───────────┘
                 │
         ┌───────▼────────┐
         │   DATABASE     │
         │  (MongoDB)     │
         │                │
         │ • Users        │
         │ • Assessments  │
         │ • History      │
         └────────────────┘
```

---

### **File Structure Explained**

```
multimodal system/
│
├── 🌐 WEB APPLICATION
│   ├── app.py                    # Main Flask application (routes, API endpoints)
│   ├── user_interface.py         # CLI interface helper
│   ├── templates/                # HTML pages
│   │   ├── index.html           # Homepage
│   │   ├── assessment.html      # Assessment form
│   │   ├── results.html         # Results display
│   │   └── ...
│   └── static/                   # CSS, JS, images
│
├── 🤖 AI & ML COMPONENTS
│   ├── pipeline.py               # Health assessment orchestrator
│   ├── train_all_models.py      # Model training script
│   ├── models/                   # Disease prediction models
│   │   ├── diabetes_model.py
│   │   ├── heart_model.py
│   │   ├── hypertension_model.py
│   │   └── obesity_model.py
│   └── saved_models/             # Trained model files (.pkl)
│
├── 📸 MULTIMODAL AGENTS
│   ├── agents/
│   │   ├── facial_agent.py      # Face analysis (MediaPipe)
│   │   ├── report_extractor.py  # PDF OCR & data extraction
│   │   ├── multimodal_scorer.py # Combined health scoring
│   │   └── xray_analyzer.py     # [DISABLED] X-ray analysis
│   └── Face feature/             # Facial recognition utilities
│
├── 🥗 NUTRITION & ANALYSIS
│   ├── nutrition_analyzer.py     # Food label scanner & analyzer
│   ├── health_scorer.py          # Health scoring algorithms
│   └── report_generator.py       # PDF report generation
│
├── 💬 WHATSAPP INTEGRATION
│   ├── whatsapp_handler.py       # Message processing logic
│   ├── whatsapp_routes.py        # Webhook routes
│   └── WHATSAPP_SETUP.md         # Setup instructions
│
├── 💾 DATABASE
│   ├── database_mongo.py         # MongoDB operations
│   └── database.py               # Database initialization
│
├── 📊 DATA
│   ├── dataset/                  # Training datasets (CSV files)
│   ├── uploads/                  # User uploaded files
│   └── reports/                  # Generated PDF reports
│
└── ⚙️ CONFIGURATION
    ├── .env                      # Environment variables (NOT in git)
    ├── .env.example              # Template for .env
    ├── requirements.txt          # Python dependencies
    ├── .gitignore               # Git ignore rules
    └── README2.md               # This file!
```

---

## 🧠 How It Works (Technical Deep Dive)

### **1. Health Assessment Pipeline**

```python
# Simplified flow
user_data = collect_user_input()  # From web form or CLI

# Step 1: Feature Mapping
mapped_features = feature_mapper.map_features(user_data)
# Converts user input to model-specific features

# Step 2: Individual Model Predictions
diabetes_risk = diabetes_model.predict(mapped_features['diabetes'])
heart_risk = heart_model.predict(mapped_features['heart'])
hypertension_risk = hypertension_model.predict(mapped_features['hypertension'])
obesity_risk = obesity_model.predict(mapped_features['obesity'])

# Step 3: Composite Scoring
composite_risk = health_scorer.calculate_weighted_risk({
    'diabetes': diabetes_risk,
    'heart': heart_risk,
    'hypertension': hypertension_risk,
    'obesity': obesity_risk
})

# Step 4: Grade & Recommendations
health_score = 100 - composite_risk
grade = assign_letter_grade(health_score)  # A+ to F
recommendations = generate_personalized_advice(risks, user_data)
```

---

### **2. Weighted Risk Calculation**

Each disease has a different impact on overall health:

```python
WEIGHTS = {
    'heart': 0.35,        # 35% - Most critical
    'diabetes': 0.25,     # 25% - High impact
    'hypertension': 0.25, # 25% - Significant
    'obesity': 0.15       # 15% - Manageable
}

composite_risk = (
    heart_risk * 0.35 +
    diabetes_risk * 0.25 +
    hypertension_risk * 0.25 +
    obesity_risk * 0.15
)
```

**Why these weights?**
- **Heart disease** is the #1 cause of death globally
- **Diabetes** has severe long-term complications
- **Hypertension** is a "silent killer"
- **Obesity** is manageable through lifestyle changes

---

### **3. Machine Learning Models**

All models use **Random Forest Classifiers**:

**Why Random Forest?**
- ✅ Handles non-linear relationships
- ✅ Robust to overfitting
- ✅ Works well with small datasets
- ✅ Provides feature importance
- ✅ No need for feature scaling

**Training Process:**
```python
# Example: Diabetes Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('dataset/diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)  # ~76%

# Save for production
joblib.dump(model, 'saved_models/diabetes_model.pkl')
```

---

### **4. Facial Expression Analysis**

Uses **MediaPipe Face Mesh** (Google's open-source library):

**Pipeline:**
```
Webcam → MediaPipe → 468 Facial Landmarks → Feature Extraction → ML Classifier
```

**Features Extracted:**
- Eye aspect ratio (EAR) - for stress detection
- Mouth aspect ratio (MAR) - for pain detection
- Eyebrow position - for anxiety/worry
- Facial symmetry - for emotion analysis

**Example Code:**
```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Process frame
results = face_mesh.process(rgb_frame)
landmarks = results.multi_face_landmarks[0]

# Calculate metrics
pain_score = calculate_pain_from_landmarks(landmarks)
stress_score = calculate_stress_from_landmarks(landmarks)
```

---

### **5. Report Extraction (OCR)**

Extracts data from PDF medical reports:

**Technology Stack:**
- **PyPDF2** - PDF text extraction
- **Regular Expressions** - Pattern matching for values
- **Custom Parsers** - Report-specific extraction logic

**Example:**
```python
import PyPDF2
import re

# Extract text from PDF
with open('blood_test.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    text = pdf_reader.pages[0].extract_text()

# Find glucose value using regex
glucose_match = re.search(r'Glucose.*?(\d{2,3})', text)
if glucose_match:
    glucose_value = int(glucose_match.group(1))
```

**Supported Patterns:**
- Blood Glucose: 70-300 mg/dL
- Cholesterol: 100-400 mg/dL
- Blood Pressure: 90/60 - 180/120 mmHg
- Hemoglobin: 10-20 g/dL

---

### **6. Nutrition Analysis**

Uses **Google Gemini Vision API** for image-to-text:

**Flow:**
```
Nutrition Label Image → Gemini Vision → Text Extraction → Parsing → Health Scoring
```

**Scoring Algorithm:**
```python
def calculate_nutrition_score(nutrients):
    score = 100  # Start with perfect score
    
    # Deduct for bad stuff
    if nutrients['sugar'] > 25:  # grams
        score -= 20
    if nutrients['sodium'] > 2000:  # mg
        score -= 15
    if nutrients['saturated_fat'] > 10:  # grams
        score -= 15
    
    # Add for good stuff
    if nutrients['fiber'] > 5:
        score += 10
    if nutrients['protein'] > 10:
        score += 10
    
    return max(0, min(100, score))  # Clamp 0-100
```

---

## 🔧 API Reference

### **Assessment API**

**Endpoint:** `POST /api/assess`

**Request Body:**
```json
{
  "age": 45,
  "gender": "Male",
  "height": 175,
  "weight": 85,
  "systolic_bp": 130,
  "diastolic_bp": 85,
  "glucose": 110,
  "cholesterol": 220,
  "smoking": "No",
  "alcohol": "Occasional",
  "physical_activity": "Moderate",
  "family_history_diabetes": "Yes",
  "family_history_heart": "No"
}
```

**Response:**
```json
{
  "success": true,
  "overall_health_score": 72,
  "health_grade": "C",
  "risk_level": "Moderate",
  "composite_risk": 28.5,
  "individual_risks": {
    "diabetes": 38.2,
    "heart": 45.7,
    "hypertension": 52.1,
    "obesity": 35.4
  },
  "recommendations": [
    "Increase physical activity to at least 150 minutes per week",
    "Reduce sodium intake to below 2,300 mg/day",
    "Monitor blood glucose levels regularly"
  ]
}
```

---

### **Nutrition Scanner API**

**Endpoint:** `POST /api/analyze-nutrition`

**Request:** Multipart form with image file

**Response:**
```json
{
  "success": true,
  "health_score": 65,
  "warnings": [
    "High sugar content: 35g per serving",
    "Contains artificial sweeteners"
  ],
  "positives": [
    "Good source of fiber: 8g",
    "Low saturated fat"
  ],
  "nutrients": {
    "calories": 250,
    "protein": 5,
    "carbs": 45,
    "sugar": 35,
    "fiber": 8,
    "fat": 8,
    "saturated_fat": 2,
    "sodium": 150
  },
  "recommendations": "Consider switching to a lower-sugar alternative"
}
```

---

### **Facial Analysis API**

**Endpoint:** `POST /api/analyze-face`

**Request:** Multipart form with image frame

**Response:**
```json
{
  "success": true,
  "pain_score": 25,
  "stress_score": 60,
  "anxiety_score": 40,
  "overall_distress": 42
}
```

---

## 📊 Database Schema

### **MongoDB Collections**

#### **Users Collection**
```javascript
{
  "_id": ObjectId("..."),
  "username": "john_doe",
  "email": "john@example.com",
  "password_hash": "$2b$12$...",  // bcrypt hash
  "created_at": ISODate("2026-01-15T10:30:00Z"),
  "last_login": ISODate("2026-02-20T14:22:00Z")
}
```

#### **Assessments Collection**
```javascript
{
  "_id": ObjectId("..."),
  "user_id": ObjectId("..."),
  "timestamp": ISODate("2026-02-20T15:00:00Z"),
  
  // Assessment results
  "overall_health_score": 72,
  "health_grade": "C",
  "risk_level": "Moderate",
  "composite_risk": 28.5,
  
  // Individual risks
  "diabetes_risk": 38.2,
  "heart_risk": 45.7,
  "hypertension_risk": 52.1,
  "obesity_risk": 35.4,
  
  // User input data
  "user_data": {
    "age": 45,
    "gender": "Male",
    "height": 175,
    "weight": 85,
    // ... all input fields
  },
  
  // Multimodal data (optional)
  "facial_data": {
    "pain_score": 25,
    "stress_score": 60,
    "anxiety_score": 40
  },
  
  "report_file": "reports/report_20260220_150000.pdf"
}
```

---

## 🛡️ Security & Privacy

### **Data Protection**

✅ **Local Processing**
- All AI models run locally (no external API calls for predictions)
- Facial analysis processed on-device
- Medical reports never leave your server

✅ **Secure Storage**
- Passwords hashed with **bcrypt** (industry standard)
- Session tokens encrypted
- MongoDB secured with authentication

✅ **HTTPS Ready**
- Configure SSL certificates for production
- Secure cookies in production mode

### **Privacy Features**

- ❌ **No data selling** - Your data is yours
- ❌ **No tracking** - No analytics or telemetry
- ✅ **User control** - Delete your data anytime
- ✅ **Minimal storage** - Only essential information saved

### **GDPR Compliance**

If deploying in EU:
- Obtain user consent for data processing
- Provide data export functionality
- Implement right-to-deletion
- Update privacy policy

---

## ⚕️ Medical & Legal Disclaimer

### **⚠️ IMPORTANT NOTICE**

This system is provided **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**.

**This is NOT:**
- ❌ A medical device
- ❌ A diagnostic tool
- ❌ A replacement for professional medical advice
- ❌ Validated for clinical use
- ❌ FDA approved or certified

**This system:**
- ℹ️ Is a software demonstration
- ℹ️ Provides educational information only
- ℹ️ Should not be used for medical decisions
- ℹ️ Has not been clinically validated
- ℹ️ May produce inaccurate results

### **Always Consult Healthcare Professionals**

For any health concerns, diagnosis, or treatment:
- 👨‍⚕️ Consult a qualified physician
- 🏥 Seek professional medical care
- 📞 Call emergency services if urgent

### **Liability**

The developers assume **NO LIABILITY** for:
- Medical decisions based on this system
- Inaccurate predictions or recommendations
- Any health outcomes related to system use
- Data loss or security breaches

**USE AT YOUR OWN RISK.**

---

## 🐛 Troubleshooting Guide

### **Common Issues & Solutions**

#### **1. Models Not Loading**

**Error:** `FileNotFoundError: saved_models/diabetes_model.pkl`

**Solution:**
```bash
python train_all_models.py
```
This trains and saves all models.

---

#### **2. MongoDB Connection Failed**

**Error:** `pymongo.errors.ServerSelectionTimeoutError`

**Solutions:**
1. Check your `.env` file has correct `MONGODB_URI`
2. Verify MongoDB Atlas IP whitelist (allow 0.0.0.0/0 for testing)
3. Check username/password in connection string
4. Test connection:
```python
from pymongo import MongoClient
client = MongoClient("your_uri_here")
client.admin.command('ping')  # Should succeed
```

---

#### **3. Gemini API Error**

**Error:** `google.api_core.exceptions.PermissionDenied: 403`

**Solutions:**
1. Verify `GEMINI_API_KEY` in `.env`
2. Check API key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Enable Gemini API in Google Cloud Console
4. Check rate limits (free tier: 60 requests/minute)

---

#### **4. WhatsApp Webhook Not Working**

**Error:** Messages not received from Twilio

**Solutions:**
1. Check Twilio console webhook URL is correct
2. Ensure Flask app is publicly accessible (use ngrok for testing)
3. Verify Twilio credentials in `.env`
4. Check webhook logs in Twilio console

**ngrok setup:**
```bash
ngrok http 5000
# Copy HTTPS URL to Twilio webhook settings
```

---

#### **5. Facial Analysis Not Working**

**Error:** Camera not detected or black screen

**Solutions:**
1. Allow camera permissions in browser
2. Use HTTPS (required for camera access)
3. Check MediaPipe installation:
```bash
pip install mediapipe opencv-python
```
4. Try different browser (Chrome works best)

---

#### **6. PDF Report Generation Fails**

**Error:** `ReportLab not found` or PDF is blank

**Solutions:**
```bash
pip install reportlab pillow
```
Check `reports/` directory exists and has write permissions

---

#### **7. Nutrition Scanner Not Reading Labels**

**Error:** "Could not extract nutrition information"

**Solutions:**
1. Ensure image is clear and well-lit
2. Label text should be readable
3. Try different angle or closer photo
4. Check Gemini API key is valid
5. Supported languages: English primarily


---

## 📈 Performance Optimization

### **Tips for Faster Processing**

1. **Model Loading**
   - Models are loaded once at startup
   - Keep `train_models=False` in production

2. **Database Queries**
   - Create indexes on frequently queried fields:
   ```javascript
   db.users.createIndex({ "username": 1 })
   db.assessments.createIndex({ "user_id": 1, "timestamp": -1 })
   ```

3. **Image Processing**
   - Compress images before upload
   - Use appropriate image formats (JPEG for photos, PNG for labels)

4. **Caching**
   - Health plans cached server-side to avoid session cookie limits
   - Clear cache periodically: `health_plan_cache.clear()`

---

## 🔮 Future Enhancements

### **Planned Features**

- [ ] **Mobile App** 
- [ ] **Continuous Data** (hospital monitoring equipment integration)
- [ ] **Medication Tracker**
- [ ] **Multi-language Support**
- [ ] **AI Chatbot** for health Q&A

### **Research Directions**

- 🧬 **Genetic Risk Factors** (SNP analysis)
- 🧠 **Mental Health Assessment** (depression, ADHD screening)
- 👁️ **Retinal Image Analysis** (diabetic retinopathy)
- 🫀 **ECG Signal Processing** (arrhythmia detection)
- 🩺 **Symptom Checker** (differential diagnosis)

---

## 📚 Additional Resources

### **Documentation**

- [WHATSAPP_SETUP.md](WHATSAPP_SETUP.md) - WhatsApp integration guide
- [Face feature/README.md](Face%20feature/README.md) - Facial analysis details
- [agents/chest_xrays/README.md](agents/chest_xrays/README.md) - X-ray module (disabled)

### **Learning Resources**

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [MongoDB University](https://university.mongodb.com/)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)


---

## 🙏 Acknowledgments

### **Built With**

- **Python** - Programming language
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **MediaPipe** - Computer vision by Google
- **MongoDB** - Database
- **Google Gemini** - Generative AI
- **Twilio** - Communication APIs
- **ReportLab** - PDF generation
- **Many more** - See `requirements.txt`

### **Datasets**

- Diabetes Dataset – [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
- Heart Disease Dataset – [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Hypertension Dataset – [Kaggle](https://www.kaggle.com/datasets/ankushpanday2/hypertension-risk-prediction-dataset)
- Obesity Dataset – [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

---

## 📊 Project Statistics

- **Lines of Code:** ~15,000+
- **Python Files:** 25+
- **ML Models:** 4
- **Supported Conditions:** 4
- **API Endpoints:** 20+
- **HTML Templates:** 15+

---

## 🎯 Project Roadmap

### **Version 1.0** (Current)
- ✅ Core health assessment
- ✅ Web interface
- ✅ Facial analysis
- ✅ Nutrition scanner
- ✅ WhatsApp bot

### **Version 1.5** (Coming Soon)
- 🔄 Mobile app
- 🔄 Voice inputs
- 🔄 Wearable integration
- 🔄 Improved UI/UX

### **Version 2.0** (Future)
- 📅 Advanced analytics
- 📅 Clinical trials

---


**Thank you for using the Multimodal Health Assessment System!**

**Remember:** This is a tool for education and exploration, not medical diagnosis. Always consult healthcare professionals for medical advice.


---