"""
ShutUpNet - Model Training
Train ML models to predict shut-up counts
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

def load_and_preprocess_data(filepath="shutup_data.csv"):
    """Load and preprocess the dataset"""
    print("📂 Loading data...")
    df = pd.read_csv(filepath)
    
    # Encode categorical variables
    le_topic = LabelEncoder()
    le_tone = LabelEncoder()
    le_mood = LabelEncoder()
    
    df['topic_encoded'] = le_topic.fit_transform(df['topic'])
    df['tone_encoded'] = le_tone.fit_transform(df['tone'])
    df['mood_encoded'] = le_mood.fit_transform(df['mood_before_convo'])
    
    # Feature engineering
    df['jokes_per_minute'] = df['jokes_made'] / df['duration_minutes']
    df['danger_score'] = (df['jokes_made'] * 0.5 + df['sarcasm_level'] * 0.3)
    
    # Save encoders
    encoders = {
        'topic': le_topic,
        'tone': le_tone,
        'mood': le_mood
    }
    
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    return df, encoders


def train_models(df):
    """Train RandomForest and XGBoost models"""
    print("\n🧠 Training models...")
    
    # Prepare features and target
    feature_cols = [
        'duration_minutes', 'jokes_made', 'sarcasm_level',
        'topic_encoded', 'tone_encoded', 'mood_encoded',
        'jokes_per_minute', 'danger_score'
    ]
    
    X = df[feature_cols]
    y = df['shutup_count']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train Random Forest
    print("  🌲 Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Train XGBoost
    print("  🚀 Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    print("\n📊 Model Evaluation:")
    
    for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
        y_pred = model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n  {name}:")
        print(f"    RMSE: {rmse:.3f}")
        print(f"    MAE: {mae:.3f}")
        print(f"    R² Score: {r2:.3f}")
    
    # Save best model (XGBoost typically performs better)
    with open('shutup_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    print("\n✅ Model saved as 'shutup_model.pkl'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Feature Importance:")
    print(feature_importance)
    
    return xgb_model, scaler, feature_importance


def create_visualizations(df, feature_importance):
    """Create fun visualizations"""
    print("\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Shutup count distribution
    axes[0, 0].hist(df['shutup_count'], bins=15, color='coral', edgecolor='black')
    axes[0, 0].set_title('Distribution of Shut-Up Counts', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Shut-Up Count')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Topic vs Shutup count
    topic_avg = df.groupby('topic')['shutup_count'].mean().sort_values(ascending=False)
    axes[0, 1].barh(range(len(topic_avg)), topic_avg.values, color='skyblue')
    axes[0, 1].set_yticks(range(len(topic_avg)))
    axes[0, 1].set_yticklabels(topic_avg.index, fontsize=9)
    axes[0, 1].set_title('Average Shut-Ups by Topic', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Average Shut-Up Count')
    
    # 3. Jokes vs Shutups
    axes[1, 0].scatter(df['jokes_made'], df['shutup_count'], alpha=0.5, color='purple')
    axes[1, 0].set_title('Jokes Made vs Shut-Up Count', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Jokes Made')
    axes[1, 0].set_ylabel('Shut-Up Count')
    
    # 4. Feature importance
    top_features = feature_importance.head(6)
    axes[1, 1].barh(range(len(top_features)), top_features['importance'].values, color='green')
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['feature'].values)
    axes[1, 1].set_title('Top Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('shutup_analysis.png', dpi=300, bbox_inches='tight')
    print("  💾 Saved visualization as 'shutup_analysis.png'")


if __name__ == "__main__":
    print("🚀 ShutUpNet Model Training\n" + "="*50)
    
    # Load and preprocess
    df, encoders = load_and_preprocess_data()
    
    # Train models
    model, scaler, feature_importance = train_models(df)
    
    # Create visualizations
    create_visualizations(df, feature_importance)
    
    print("\n" + "="*50)
    print("✅ Training complete! Ready for predictions.")
    print("\n💡 Next steps:")
    print("  1. Run 'python app.py' to start the web app")
    print("  2. Or use 'python predict.py' for CLI predictions")
