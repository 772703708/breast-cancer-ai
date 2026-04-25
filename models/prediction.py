from app import db

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    age = db.Column(db.Float)
    bmi = db.Column(db.Float)
    result = db.Column(db.Float)