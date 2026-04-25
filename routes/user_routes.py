from flask import Blueprint, render_template, request
from flask_login import login_required, current_user
from ml.predict import predict
from models.prediction import Prediction
from app import db

user = Blueprint('user', __name__)

@user.route("/")
def home():
    return render_template("index.html")


@user.route("/dashboard")
@login_required
def dashboard():
    return render_template("user/dashboard.html")


@user.route("/predict", methods=["GET", "POST"])
@login_required
def prediction():
    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])

        result = predict([age, bmi])

        record = Prediction(
            user_id=current_user.id,
            age=age,
            bmi=bmi,
            result=result
        )
        db.session.add(record)
        db.session.commit()

        return render_template("user/result.html", result=result)

    return render_template("user/predict.html")


@user.route("/history")
@login_required
def history():
    data = Prediction.query.filter_by(user_id=current_user.id).all()
    return render_template("user/history.html", data=data)