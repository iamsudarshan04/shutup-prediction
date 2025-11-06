# 🎯 ShutUpNet - Shut-Up Prediction Model

A playful machine learning project that predicts how many times your girlfriend will say "shut up" during a conversation based on various factors like topic, jokes, sarcasm, and mood! 😂

## 📁 Project Structure

```
shutup/
├── generate_data.py      # Synthetic data generator with relationship dynamics
├── train_model.py        # Model training (Random Forest + XGBoost)
├── predict.py           # Command-line prediction tool
├── app.py               # Flask web application
├── templates/
│   ├── index.html       # Main web interface with danger meter
│   └── error.html       # Error page
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

This creates a synthetic dataset with 200 conversation samples based on realistic relationship dynamics:

```bash
python generate_data.py
```

**Output:**
- `shutup_data.csv` - Training dataset
- Displays dataset statistics and shut-up count distribution

### 3. Train the Model

Trains Random Forest and XGBoost models, evaluates performance, and creates visualizations:

```bash
python train_model.py
```

**Output:**
- `shutup_model.pkl` - Trained XGBoost model
- `scaler.pkl` - Feature scaler
- `encoders.pkl` - Label encoders
- `shutup_analysis.png` - Visualizations (distribution, topics, feature importance)

### 4. Make Predictions

#### Option A: Web Application (Recommended)

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

Features:
- 🎨 Beautiful gradient UI
- 📊 Real-time danger meter
- 🎯 Visual prediction results
- 💡 Helpful advice based on risk level

#### Option B: Command Line

```bash
python predict.py
```

Interactive CLI that prompts for:
- Topic of conversation
- Duration
- Number of jokes
- Sarcasm level
- Your tone
- Her mood

## 📊 Features

The model uses these features to make predictions:

1. **Topic** - What you're talking about (e.g., "her friends", "teasing her looks", "saying scene")
2. **Duration** - Conversation length in minutes
3. **Jokes Made** - Number of jokes/teasing comments
4. **Sarcasm Level** - Scale of 0-5
5. **Tone** - Your conversational tone (funny, serious, calm, etc.)
6. **Mood** - Her mood before the conversation (happy, tired, annoyed, etc.)

### Engineered Features

- **Jokes per Minute** - Intensity of joking
- **Danger Score** - Composite risk score

## 🎯 Behavioral Rules (Built into Data Generator)

Based on real relationship dynamics:

- ✅ **"Scene" keyword** → Always safe (0 shut-ups)
- ⚠️ **Talking about her** → High risk (+2-4 shut-ups)
- 🔥 **Making jokes (>3)** → Very risky (+1-3 shut-ups)
- 😏 **High sarcasm (>3)** → Adds fuel (+1-2 shut-ups)
- 🎭 **Funny tone** → Slight increase (+1 shut-up)
- 😴 **Tired/annoyed mood** → Amplifies risk (+1-2 shut-ups)
- 💬 **Long conversations (>90 min)** → Slight risk increase

## 📈 Model Performance

The XGBoost model achieves:
- **RMSE**: ~0.5-0.8 shut-ups
- **MAE**: ~0.4-0.6 shut-ups
- **R² Score**: ~0.75-0.85

Top important features:
1. Danger Score (composite)
2. Jokes Made
3. Topic
4. Sarcasm Level

## 🎨 Web App Features

### Danger Levels

| Level | Count | Color | Meaning |
|-------|-------|-------|---------|
| ✅ SAFE ZONE | 0 | Green | Smooth sailing! |
| 😌 Low Risk | 1-2 | Light Green | Proceed with caution |
| ⚠️ Moderate Risk | 3-4 | Yellow | Tread carefully |
| 🚨 High Risk | 5-6 | Orange | Danger ahead! |
| 💀 DANGER ZONE | 7+ | Red | Abort mission! |

### Visual Danger Meter

The web app includes an animated danger meter that fills up based on the predicted risk level, with color-coded warnings and personalized advice.

## 🤓 Example Predictions

**Safe Conversation:**
```
Topic: saying scene
Duration: 60 minutes
Jokes: 0
Sarcasm: 1
Tone: calm
Mood: happy
→ Prediction: 0 shut-ups ✅
```

**Dangerous Conversation:**
```
Topic: teasing her looks
Duration: 70 minutes
Jokes: 6
Sarcasm: 4
Tone: funny
Mood: tired
→ Prediction: 7+ shut-ups 💀
```

## ⚠️ Disclaimer

This is a **fun, playful project** meant for entertainment and learning! 

- Always communicate respectfully in real relationships
- Make sure your partner is aware and okay with this lighthearted experiment
- Don't actually use this to manipulate conversations (that would be weird 😅)
- Results are based on synthetic data and may not reflect reality

## 🛠️ Tech Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn, XGBoost
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## 📝 Future Enhancements

Potential improvements:
- [ ] Add real conversation logging (with consent)
- [ ] Implement NLP for text sentiment analysis
- [ ] Add time-of-day patterns
- [ ] Create mobile app version
- [ ] Add conversation history tracking
- [ ] Implement fine-tuning based on actual results

## 🎉 Have Fun!

Remember: The best prediction model is good communication and mutual respect! This project is just for laughs and to practice some data science skills. 😄

---

**Created with ❤️ and data science**
