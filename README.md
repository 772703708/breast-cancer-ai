# 🧠 Breast Cancer Prediction Model

## 📌 Overview

هذا المشروع يهدف إلى بناء نموذج تعلم آلي للتنبؤ بسرطان الثدي باستخدام بيانات طبية، مع واجهة تطبيق تستقبل مدخلات المستخدم وتعرض مستوى الخطورة.

---

## 🔗 Dataset & Notebook

* 📊 رابط البيانات والكود:
  https://www.kaggle.com/code/dhainjeamita/breast-cancer-dataset-classification/notebook

---

## ⚙️ بيئة التدريب والأدوات المستخدمة

تم تدريب النموذج باستخدام **Google Colab** (بيئة سحابية تدعم GPU).

### 📚 المكتبات المستخدمة:

| المكتبة                 | الاستخدام             |
| ----------------------- | --------------------- |
| pandas                  | قراءة وتحليل البيانات |
| numpy                   | العمليات الحسابية     |
| sklearn.model_selection | تقسيم البيانات        |
| sklearn.preprocessing   | توحيد القيم           |
| sklearn.linear_model    | Logistic Regression   |
| sklearn.metrics         | تقييم النموذج         |
| joblib                  | حفظ النموذج           |

---

## 🧪 خطوات تدريب النموذج

### 1️⃣ تحميل البيانات

```python
df = pd.read_csv('data.csv')
```

---

### 2️⃣ تنظيف البيانات

```python
df = df.drop('id', axis=1)
df = df.drop('Unnamed: 32', axis=1)
```

---

### 3️⃣ تحويل القيم

```python
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```

---

### 4️⃣ فصل البيانات

```python
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
```

---

### 5️⃣ تقسيم البيانات

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

### 6️⃣ توحيد القيم

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 7️⃣ تدريب النموذج

```python
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
```

---

### 8️⃣ تقييم النموذج

```python
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
```

---

### 9️⃣ حفظ النموذج

```python
joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
```

---

## 🧩 شرح دوال التطبيق

### 🔹 تحميل النموذج

```python
def load_ml_models():
    global model, scaler, feature_columns, model_loaded
    model = joblib.load('ml/breast_cancer_model.pkl')
    scaler = joblib.load('ml/scaler.pkl')
    feature_columns = joblib.load('ml/feature_columns.pkl')
    model_loaded = True
```

---

### 🔹 حساب المخاطر

```python
def calculate_risk_score(age, bmi, tumor_size, inv_nodes, metastasis, history, menopause, breast_side):
    risk_score = 0.0
    if age >= 60: risk_score += 0.20
    if bmi >= 30: risk_score += 0.15
    if tumor_size >= 30: risk_score += 0.18
    if metastasis == 'yes': risk_score += 0.25
    return min(risk_score, 1.0)
```

---

### 🔹 إنشاء الميزات

```python
def create_full_feature_vector(form_data):
    features = np.zeros(30)
    risk_score = calculate_risk_score(...)
    features[0] = 10 + (tumor_size * 0.1) + (risk_score * 4)
    features[6] = 0.04 + (risk_score * 0.12)
    return features.reshape(1, -1)
```

---

### 🔹 تصنيف مستوى الخطورة

```python
def get_risk_level(probability):
    prob_percent = probability * 100
    if prob_percent >= 85: return 'Very High Risk'
    elif prob_percent >= 70: return 'High Risk'
    elif prob_percent >= 55: return 'Moderate to High Risk'
    elif prob_percent >= 40: return 'Moderate Risk'
    elif prob_percent >= 25: return 'Low to Moderate Risk'
    elif prob_percent >= 10: return 'Low Risk'
    else: return 'Very Low Risk'
```

---

### 🔹 دالة التنبؤ

```python
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        form_data = {
            'age': request.form.get('age'),
            'bmi': request.form.get('bmi'),
        }
        features = create_full_feature_vector(form_data)
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        risk_info = get_risk_level(probability)
```

---

## 🔄 تدفق البيانات

```
مدخلات المستخدم
↓
create_full_feature_vector
↓
30 Features
↓
Scaler
↓
Model Prediction
↓
Probability
↓
Risk Level
```

---

## 📊 نتائج النموذج

| المقياس            | القيمة |
| ------------------ | ------ |
| عدد عينات التدريب  | 455    |
| عدد عينات الاختبار | 114    |
| عدد الميزات        | 30     |
| الدقة              | 97.37% |
| ROC-AUC            | 99.50% |
| حجم النموذج        | 1.3KB  |

---

## ⚠️ التحديات

* عدم توازن البيانات → استخدام Stratify
* اختلاف القيم → StandardScaler
* تحويل المدخلات → Feature Engineering
* تصنيف النتائج → 7 مستويات خطورة

---

## 👨‍💻 Author

تم تطوير هذا المشروع لأغراض تعليمية وبحثية في مجال الذكاء الاصطناعي والتطبيقات الطبية.
