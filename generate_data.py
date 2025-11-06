"""
ShutUpNet - Synthetic Data Generator
Generates realistic conversation data based on girlfriend relationship dynamics
"""

import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define possible values for each feature
TOPICS = [
    "her friends", "her behavior", "about her", "saying scene", 
    "serious life talk", "random memes", "her work stress", 
    "teasing her looks", "planning trip", "teasing her behavior",
    "making plans", "discussing movie", "talking about food",
    "complaining about something", "asking for advice", "general chat"
]

TONES = ["funny", "chill", "serious", "calm", "sarcastic"]
MOODS = ["happy", "neutral", "tired", "excited", "annoyed", "stressed"]


def calculate_shutup_count(topic, jokes_made, sarcasm_level, tone, mood, duration):
    """
    Calculate shut-up count based on behavioral rules
    """
    shutup_count = 0
    
    # Rule 1: "scene" keyword is always safe
    if "scene" in topic.lower():
        return 0
    
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
    if jokes_made > 3:
        shutup_count += random.randint(1, 3)
    elif jokes_made > 5:
        shutup_count += random.randint(2, 4)
    
    # Rule 4: Sarcasm adds fuel to fire
    if sarcasm_level > 3:
        shutup_count += random.randint(1, 2)
    
    # Rule 5: Tone matters
    if tone == "funny" or tone == "sarcastic":
        shutup_count += 1
    elif tone == "serious" or tone == "calm":
        shutup_count = max(0, shutup_count - 1)
    
    # Rule 6: Mood amplifier
    if mood == "annoyed" or mood == "tired":
        shutup_count += random.randint(1, 2)
    elif mood == "happy" or mood == "excited":
        shutup_count = max(0, shutup_count - random.randint(0, 1))
    
    # Rule 7: Duration can increase chances (slightly)
    if duration > 90:
        if random.random() > 0.7:  # 30% chance
            shutup_count += 1
    
    # Add some randomness to make it realistic
    shutup_count += random.randint(-1, 1)
    
    # Ensure non-negative
    return max(0, shutup_count)


def generate_dataset(num_samples=200):
    """
    Generate synthetic conversation dataset
    """
    data = []
    
    for _ in range(num_samples):
        topic = random.choice(TOPICS)
        duration = random.randint(30, 120)  # 30 min to 2 hours
        tone = random.choice(TONES)
        mood = random.choice(MOODS)
        
        # Generate jokes and sarcasm based on tone
        if tone == "funny" or tone == "sarcastic":
            jokes_made = random.randint(3, 8)
            sarcasm_level = random.randint(2, 5)
        elif tone == "serious" or tone == "calm":
            jokes_made = random.randint(0, 2)
            sarcasm_level = random.randint(0, 1)
        else:
            jokes_made = random.randint(1, 4)
            sarcasm_level = random.randint(1, 3)
        
        # Calculate shut-up count
        shutup_count = calculate_shutup_count(
            topic, jokes_made, sarcasm_level, tone, mood, duration
        )
        
        data.append({
            "topic": topic,
            "duration_minutes": duration,
            "jokes_made": jokes_made,
            "sarcasm_level": sarcasm_level,
            "tone": tone,
            "mood_before_convo": mood,
            "shutup_count": shutup_count
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("🎭 Generating ShutUpNet training data...")
    
    # Generate dataset
    df = generate_dataset(200)
    
    # Save to CSV
    df.to_csv("shutup_data.csv", index=False)
    
    print(f"✅ Generated {len(df)} conversation samples")
    print("\n📊 Dataset Overview:")
    print(df.head(10))
    print("\n📈 Statistics:")
    print(df.describe())
    print("\n🎯 Shut-up Count Distribution:")
    print(df["shutup_count"].value_counts().sort_index())
    print(f"\n💾 Saved to: shutup_data.csv")
