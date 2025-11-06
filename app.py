"""
ShutUpNet - Flask Web Application
Fun web interface with danger meter for shut-up predictions
Uses rule-based prediction system (no ML dependencies!)
"""

from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# Available options for dropdowns
TOPICS = [
    "her friends", "her behavior", "about her", "saying scene", 
    "serious life talk", "random memes", "her work stress", 
    "teasing her looks", "planning trip", "teasing her behavior",
    "making plans", "discussing movie", "talking about food",
    "complaining about something", "asking for advice", "general chat"
]

TONES = ["funny", "chill", "serious", "calm", "sarcastic"]
MOODS = ["happy", "neutral", "tired", "excited", "annoyed", "stressed"]


def predict_shutup(topic, duration, jokes, sarcasm, tone, mood):
    """
    Rule-based prediction system based on relationship dynamics
    No ML models needed!
    """
    try:
        shutup_count = 0
        
        # Rule 1: "scene" keyword is always safe
        if "scene" in topic.lower():
            shutup_count = 0
        else:
            # Rule 2: Baseline based on topic
            if "her" in topic or "about her" in topic:
                shutup_count += random.randint(2, 4)
            elif "teasing" in topic:
                shutup_count += random.randint(3, 5)
            elif "serious" in topic or "work" in topic:
                shutup_count += random.randint(0, 1)
            elif "friends" in topic:
                shutup_count += random.randint(1, 3)
            else:
                shutup_count += random.randint(0, 2)
            
            # Rule 3: Jokes are risky!
            if jokes > 5:
                shutup_count += random.randint(2, 4)
            elif jokes > 3:
                shutup_count += random.randint(1, 3)
            
            # Rule 4: Sarcasm adds fuel to fire
            if sarcasm > 3:
                shutup_count += random.randint(1, 2)
            
            # Rule 5: Tone matters
            if tone == "funny" or tone == "sarcastic":
                shutup_count += 1
            elif tone == "serious" or tone == "calm":
                shutup_count = max(0, shutup_count - 1)
            
            # Rule 6: Mood amplifier
            if mood == "annoyed" or mood == "tired" or mood == "stressed":
                shutup_count += random.randint(1, 2)
            elif mood == "happy" or mood == "excited":
                shutup_count = max(0, shutup_count - random.randint(0, 1))
            
            # Rule 7: Duration can increase chances (slightly)
            if duration > 90:
                if random.random() > 0.7:  # 30% chance
                    shutup_count += 1
            
            # Add some natural randomness
            shutup_count += random.randint(-1, 1)
        
        # Ensure non-negative
        count = max(0, shutup_count)
        
        # Determine danger level
        if count == 0:
            danger_level = "SAFE ZONE"
            danger_emoji = "✅"
            danger_color = "#28a745"
            advice = "Safe to proceed! This conversation should go smoothly."
            danger_percentage = 0
        elif count <= 2:
            danger_level = "Low Risk"
            danger_emoji = "😌"
            danger_color = "#90ee90"
            advice = "Proceed with caution, but you should be fine."
            danger_percentage = 25
        elif count <= 4:
            danger_level = "Moderate Risk"
            danger_emoji = "⚠️"
            danger_color = "#ffc107"
            advice = "Be careful! Consider reducing jokes."
            danger_percentage = 50
        elif count <= 6:
            danger_level = "High Risk"
            danger_emoji = "🚨"
            danger_color = "#ff8c00"
            advice = "Proceed at your own risk. Tread carefully!"
            danger_percentage = 75
        else:
            danger_level = "DANGER ZONE"
            danger_emoji = "💀"
            danger_color = "#dc3545"
            advice = "ABORT MISSION! Maybe talk about 'scene' instead? 😅"
            danger_percentage = 100
        
        return {
            "shutup_count": count,
            "danger_level": danger_level,
            "danger_emoji": danger_emoji,
            "danger_color": danger_color,
            "advice": advice,
            "danger_percentage": danger_percentage
        }
    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', 
                         topics=TOPICS, 
                         tones=TONES, 
                         moods=MOODS)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = request.json
        
        topic = data.get('topic')
        duration = int(data.get('duration', 60))
        jokes = int(data.get('jokes', 0))
        sarcasm = int(data.get('sarcasm', 0))
        tone = data.get('tone')
        mood = data.get('mood')
        
        result = predict_shutup(topic, duration, jokes, sarcasm, tone, mood)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ShutUpNet Web Application Starting...")
    print("="*60)
    print("\n✅ Rule-based prediction engine ready!")
    print("🌐 Access the app at: http://localhost:5000")
    print("\n💡 Using smart behavioral rules (no ML training needed!)")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
