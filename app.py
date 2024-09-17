# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
import json
import os

app = Flask(__name__)

# Replace with your actual API key
USDA_API_KEY = 'mbBOa2HVHlzbokQfuEAMp9gu75gxUfJcSroFrRlT'

class AIDietician:
    def __init__(self):
        self.load_data()
        self.train_model()

    def load_data(self):
        # Load data from USDA FoodData Central API
        api_url = f'https://api.nal.usda.gov/fdc/v1/foods/search?api_key={USDA_API_KEY}'
        
        # Define categorized food lists
        self.food_categories = {
            'veg': ['apple', 'banana', 'broccoli', 'rice', 'spinach', 'carrot'],
            'non_veg': ['chicken breast', 'salmon', 'egg', 'milk'],
            'vegan': ['apple', 'banana', 'broccoli', 'rice', 'spinach', 'carrot', 'almonds', 'soy milk']
        }
        
        # Initialize an empty DataFrame for food data
        self.food_data = pd.DataFrame()
        
        for category, foods in self.food_categories.items():
            for food in foods:
                params = {
                    'query': food,
                    'dataType': 'Survey (FNDDS)',
                    'pageSize': 1
                }
                response = requests.get(api_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data['foods']:
                        food_item = data['foods'][0]
                        food_item['category'] = category
                        # Convert the food_item dictionary to a DataFrame
                        food_item_df = pd.DataFrame([food_item])
                        # Concatenate with the existing food_data DataFrame
                        self.food_data = pd.concat([self.food_data, food_item_df], ignore_index=True)
        
        # For simplicity, we'll create a synthetic dataset for diet plans
        np.random.seed(42)
        n_samples = 1000
        
        self.data = pd.DataFrame({
            'age': np.random.randint(18, 70, n_samples),
            'weight': np.random.randint(45, 120, n_samples),
            'height': np.random.randint(150, 200, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'activity_level': np.random.choice(['Sedentary', 'Lightly active', 'Moderately active', 'Very active', 'Super active'], n_samples),
            'diet_plan': np.random.choice(['Low Carb', 'Balanced', 'High Protein', 'Vegetarian', 'Vegan'], n_samples),
            'meal_preference': np.random.choice(['Veg', 'Non-Veg', 'Vegan'], n_samples)
        })
        
        le = LabelEncoder()
        self.data['gender'] = le.fit_transform(self.data['gender'])
        self.data['activity_level'] = le.fit_transform(self.data['activity_level'])
        self.data['diet_plan'] = le.fit_transform(self.data['diet_plan'])
        self.data['meal_preference'] = le.fit_transform(self.data['meal_preference'])
        
        self.diet_plan_encoder = dict(zip(le.classes_, le.transform(le.classes_)))
        self.diet_plan_decoder = dict(zip(le.transform(le.classes_), le.classes_))
        
        self.meal_preference_encoder = dict(zip(['Veg', 'Non-Veg', 'Vegan'], range(3)))
        self.meal_preference_decoder = dict(zip(range(3), ['Veg', 'Non-Veg', 'Vegan']))

    def train_model(self):
        X = self.data[['age', 'weight', 'height', 'gender', 'activity_level', 'meal_preference']]
        y = self.data['diet_plan']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        print(f"Model accuracy: {self.model.score(X_test, y_test):.2f}")

    def generate_diet_plan(self, age, weight, height, gender, activity_level, meal_preference):
        gender_encoded = 0 if gender == 'Female' else 1
        activity_encoded = ['Sedentary', 'Lightly active', 'Moderately active', 'Very active', 'Super active'].index(activity_level)
        meal_pref_encoded = self.meal_preference_encoder[meal_preference]
        
        bmi = weight / ((height/100) ** 2)
        
        if gender == 'Female':
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        else:
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        
        activity_factors = [1.2, 1.375, 1.55, 1.725, 1.9]
        tdee = bmr * activity_factors[activity_encoded]
        
        prediction = self.model.predict([[age, weight, height, gender_encoded, activity_encoded, meal_pref_encoded]])[0]
        diet_plan = self.diet_plan_decoder[prediction]
        
        meal_plan = self.get_meal_plan(diet_plan, tdee, meal_preference)
        
        return {
            'bmi': round(bmi, 2),
            'tdee': round(tdee, 2),
            'diet_plan': diet_plan,
            'meal_plan': meal_plan
        }

    def get_meal_plan(self, diet_plan, tdee, meal_preference):
        # Filter food data based on meal preference
        food_data_filtered = self.food_data[self.food_data['category'] == meal_preference.lower()]
        
        # Create meal plan based on filtered food data
        meal_plan = f"Daily Calorie Target: {tdee:.0f}\n\n"
        
        breakfast = food_data_filtered.sample(1)  # Just pick one item for simplicity
        lunch = food_data_filtered.sample(1)
        dinner = food_data_filtered.sample(1)
        snack = food_data_filtered.sample(1)
        
        meal_plan += f"Breakfast:\n- {breakfast['description'].values[0]}\n\n"
        meal_plan += f"Lunch:\n- {lunch['description'].values[0]}\n\n"
        meal_plan += f"Dinner:\n- {dinner['description'].values[0]}\n\n"
        meal_plan += f"Snack:\n- {snack['description'].values[0]}"
        
        return meal_plan

dietician = AIDietician()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    data = request.json
    result = dietician.generate_diet_plan(
        int(data['age']),
        float(data['weight']),
        float(data['height']),
        data['gender'],
        data['activity_level'],
        data['meal_preference']
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
