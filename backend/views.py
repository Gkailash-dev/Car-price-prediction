from django.shortcuts import render
from django.views import View
from django.conf import settings
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


class home(View):
    def get(self, request):
        return render(request, "index.html", {"prediction": ""})

    def post(self, request): 
        file_path = os.path.join(settings.BASE_DIR, 'frontend', 'data', 'heart.csv')
        df = pd.read_csv(file_path)

    # Get user inputs
        d1 = request.POST.get("brand", "").strip().lower()
        d2 = float(request.POST.get("make_year", 0))
        d3 = request.POST.get("transmission", "").strip().lower()
        d4 = request.POST.get("fuel_type", "").strip().lower()
        d5 = float(request.POST.get("engine_cc", 0))
        d6 = float(request.POST.get("km_driven", 0))
        d7 = request.POST.get("ownership", "").strip().lower()
    
    # ... rest of your ML pipeline ...

        # Load dataset
        
        # Encode categoricals (basic encoding — adjust for your dataset)
        d1 = 1 if d1 == "maruti" else 0
        d3 = 1 if d3 == "manual" else 0
        d4 = 1 if d4 == "diesel" else 0
        d7 = 1 if d7 == "1st owner" else 0

                # Fix column names
        df.rename(columns={'engine_capacity(CC)': 'engine_cc'}, inplace=True)

        # Encode categoricals in dataset
        df = pd.get_dummies(df, columns=['brand', 'transmission', 'fuel_type', 'ownership'], drop_first=True)

        # Features and target
        # Features and target
        X = df.drop(['price'], axis=1)
        y = df['price']

# ✅ Fix missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

# Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ✅ Collect user inputs
        user_input = {
            'make_year': float(request.POST.get("make_year", 0)),
            'engine_cc': float(request.POST.get("engine_cc", 0)),
            'km_driven': float(request.POST.get("km_driven", 0)),
            # Dummy placeholders for categoricals (all set to 0 initially)
        }

        # Create DataFrame with numeric features
        user_input_df = pd.DataFrame([user_input])

        # Add missing dummy columns (same as training)
        for col in X.columns:
            if col not in user_input_df.columns:
                user_input_df[col] = 0

        # Reorder columns to match training
        user_input_df = user_input_df[X.columns]

        # Scale user input
        user_input_scaled = scaler.transform(user_input_df)

        # Train model
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)

        # Predict
        prediction = lr.predict(user_input_scaled)
        p = round(prediction[0], 2)

        return render(request, 'index.html', {"prediction": p})
