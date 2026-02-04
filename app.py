from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# ---------------- LOAD OBJECTS ----------------
with open("best_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("cat_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("feature_column.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("Churn_Prediction_Best_Model.pkl", "rb") as f:
    model = pickle.load(f)


# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html", selected_sim="")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # GET FORM DATA
        data = {
            "gender": request.form["gender"],
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form.get("MultipleLines", "No"),
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form.get("OnlineSecurity", "No"),
            "OnlineBackup": request.form.get("OnlineBackup", "No"),
            "DeviceProtection": request.form.get("DeviceProtection", "No"),
            "TechSupport": request.form.get("TechSupport", "No"),
            "StreamingTV": request.form.get("StreamingTV", "No"),
            "StreamingMovies": request.form.get("StreamingMovies", "No"),
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form.get("PaperlessBilling", "No"),
            "PaymentMethod": request.form["PaymentMethod"],
            "SIM": request.form["SIM"],
            "DeviceType": request.form.get("DeviceType", "Old Device"),
            "Region": request.form.get("Region", "Urban"),
            "tenure": float(request.form["tenure"]),
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"]),
        }

        df = pd.DataFrame([data])

        # ENCODE CATEGORICALS
        cat_cols = df.select_dtypes(include="object").columns
        df[cat_cols] = encoder.transform(df[cat_cols])

        # ALIGN FEATURES
        df = df.reindex(columns=feature_columns, fill_value=0)

        # SCALE NUMERICALS
        df_scaled = scaler.transform(df)

        # PREDICT
        prediction = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]

        # FORMAT RESULT
        result = "Churn" if prediction == 1 else "Not Churn"
        probability = f"{prob * 100:.2f}%"

        # DETERMINE RISK LEVEL
        prob_percent = prob * 100
        if prob_percent >= 60:
            risk = "High Risk 🔴"
        elif prob_percent >= 30:
            risk = "Medium Risk 🟡"
        else:
            risk = "Low Risk 🟢"

        return render_template(
            "index.html",
            prediction=result,
            probability=probability,
            risk=risk,
            selected_sim=data["SIM"]
        )

    except Exception as e:
        return render_template("index.html", error=str(e), selected_sim=request.form.get("SIM", ""))


if __name__ == "__main__":
    app.run(debug=True)
