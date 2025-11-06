"""
ShutUpNet - Command Line Predictor
Quick predictions from the terminal
"""

import pickle
import numpy as np
import pandas as pd

def load_models():
    """Load trained model and preprocessors"""
    with open('shutup_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders


def get_danger_level(count):
    """Get danger level emoji and text"""
    if count == 0:
        return "✅ SAFE ZONE", "green"
    elif count <= 2:
        return "😌 Low Risk", "lightgreen"
    elif count <= 4:
        return "⚠️ Moderate Risk", "yellow"
    elif count <= 6:
        return "🚨 High Risk", "orange"
    else:
        return "💀 DANGER ZONE", "red"


def predict_shutup(topic, duration, jokes, sarcasm, tone, mood):
    """Make a prediction"""
    model, scaler, encoders = load_models()
    
    # Encode inputs
    topic_encoded = encoders['topic'].transform([topic])[0]
    tone_encoded = encoders['tone'].transform([tone])[0]
    mood_encoded = encoders['mood'].transform([mood])[0]
    
    # Calculate engineered features
    jokes_per_minute = jokes / duration
    danger_score = jokes * 0.5 + sarcasm * 0.3
    
    # Prepare feature vector
    features = np.array([[
        duration, jokes, sarcasm,
        topic_encoded, tone_encoded, mood_encoded,
        jokes_per_minute, danger_score
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    return max(0, round(prediction))


def interactive_predict():
    """Interactive prediction interface"""
    print("\n" + "="*60)
    print("🎯 ShutUpNet - Shut-Up Prediction System")
    print("="*60 + "\n")
    
    # Load models to get available options
    _, _, encoders = load_models()
    
    print("📋 Available Topics:")
    topics = list(encoders['topic'].classes_)
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic}")
    
    topic_idx = int(input("\n🔹 Select topic number: ")) - 1
    topic = topics[topic_idx]
    
    duration = int(input("🔹 Conversation duration (minutes): "))
    jokes = int(input("🔹 Number of jokes made: "))
    sarcasm = int(input("🔹 Sarcasm level (0-5): "))
    
    print("\n📋 Available Tones:", ", ".join(encoders['tone'].classes_))
    tone = input("🔹 Tone: ")
    
    print("📋 Available Moods:", ", ".join(encoders['mood'].classes_))
    mood = input("🔹 Her mood before convo: ")
    
    # Predict
    count = predict_shutup(topic, duration, jokes, sarcasm, tone, mood)
    danger_level, color = get_danger_level(count)
    
    print("\n" + "="*60)
    print("🎯 PREDICTION RESULT")
    print("="*60)
    print(f"\n  Expected 'Shut Ups': {count}")
    print(f"  Danger Level: {danger_level}")
    print("\n" + "="*60)
    
    # Fun advice
    if count == 0:
        print("\n💚 Safe to proceed! This conversation should go smoothly.")
    elif count <= 2:
        print("\n💛 Proceed with caution, but you should be fine.")
    elif count <= 4:
        print("\n🧡 Be careful! Consider reducing jokes.")
    else:
        print("\n❤️ ABORT MISSION! Maybe talk about 'scene' instead? 😅")


if __name__ == "__main__":
    try:
        interactive_predict()
    except FileNotFoundError:
        print("\n❌ Error: Models not found!")
        print("Please run 'python generate_data.py' and 'python train_model.py' first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
