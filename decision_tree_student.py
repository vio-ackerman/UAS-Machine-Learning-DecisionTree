import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ================================
# 1. LOAD DATASET
# ================================
data = pd.read_csv("student-mat.csv", sep=';')

print("Jumlah data:", data.shape)
print(data.head())

# ================================
# 2. PREPROCESSING DATA
# ================================

# Membuat target klasifikasi
# Lulus (1) jika nilai akhir >= 10, Tidak lulus (0) jika < 10
data['performance'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Menghapus kolom nilai asli
data.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)

# Encoding data kategorikal
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = encoder.fit_transform(data[col])

# ================================
# 3. PEMBAGIAN DATA
# ================================
X = data.drop('performance', axis=1)
y = data['performance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data training:", X_train.shape)
print("Data testing :", X_test.shape)

# ================================
# 4. MEMBANGUN MODEL DECISION TREE
# ================================
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# 5. EVALUASI MODEL
# ================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================================
# 6. VISUALISASI POHON KEPUTUSAN
# ================================
plt.figure(figsize=(22, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Tidak Lulus', 'Lulus'],
    filled=True
)
plt.title("Decision Tree - Student Performance")
plt.show()
