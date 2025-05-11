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
            # Get user inputs
            user_input = {
                'category': request.form['category'],
                'budget_level': request.form['budget_level'],
                'season': request.form['season'],
                'trip_type': request.form['trip_type'],
                'duration_days': float(request.form['duration_days'])
            }

            # Encode and scale input
            encoded = [encoders[col].transform([user_input[col]])[0] for col in ['category', 'budget_level', 'season', 'trip_type']]
            scaled_duration = scaler.transform([[user_input['duration_days']]])[0][0]
            features = np.array(encoded + [scaled_duration]).reshape(1, -1)

            # Predict the cluster
            cluster = model.predict(features)[0]

            # Select 5 unique places from predicted cluster
            matches = df[df['cluster'] == cluster].drop_duplicates(subset='place_name').sample(n=5)
            recommendations = matches[['place_name', 'description']].to_dict(orient='records')

            return render_template('result.html', recommendations=recommendations)

        except Exception as e:
            return f"<h3>Error: {str(e)}</h3>"

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)