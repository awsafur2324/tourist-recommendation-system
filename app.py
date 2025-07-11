
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model assets from the model folder
model = joblib.load(os.path.join('model', 'final_model.pkl'))
scaler = joblib.load(os.path.join('model', 'scaler.pkl'))
encoders = joblib.load(os.path.join('model', 'label_encoders.pkl'))
df = joblib.load(os.path.join('model', 'full_df_with_descriptions.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            # Get user input
            user_input = {
                'category': request.form['category'],
                'budget_level': request.form['budget_level'],
                'season': request.form['season'],
                'trip_type': request.form['trip_type'],
                'duration_days': float(request.form['duration_days'])
            }

            # Encode + scale
            encoded = [encoders[col].transform([user_input[col]])[0] for col in ['category', 'budget_level', 'season', 'trip_type']]
            scaled_duration = scaler.transform([[user_input['duration_days']]])[0][0]
            features = np.array(encoded + [scaled_duration]).reshape(1, -1)

            # Predict cluster
            cluster = model.predict(features)[0]

            # Filter cluster
            cluster_df = df[df['cluster'] == cluster]

            # Filter by exact category match
            cat_encoded = encoders['category'].transform([user_input['category']])[0]
            strict_match = cluster_df[cluster_df['category'] == cat_encoded].drop_duplicates(subset='place_name')

            # Fallback: if not enough, show similar category places separately
            additional_suggestions = []
            if len(strict_match) < 5:
                others = cluster_df[cluster_df['category'] != cat_encoded].drop_duplicates(subset='place_name')
                additional_suggestions = others.head(5 - len(strict_match))[
                    ['place_name', 'description']
                ].to_dict(orient='records')

            # Final output
            strict_recommendations = strict_match[['place_name', 'description']].to_dict(orient='records')

            return render_template(
                'result.html',
                strict_recommendations=strict_recommendations,
                additional_suggestions=additional_suggestions,
                selected_category=user_input['category']
            )

        except Exception as e:
            return f"<h3>Error: {str(e)}</h3>"

    return render_template('form.html')
if __name__ == '__main__':
    app.run(debug=True)