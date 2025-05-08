import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dữ liệu
train = np.load("train_embeddings.npz")
X_train, y_train = train["X"], train["y"]

test = np.load("test_embeddings.npz")
X_test, y_test = test["X"], test["y"]

# Huấn luyện SVM
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, "model_svm.joblib")

# Huấn luyện Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
joblib.dump(rf, "model_rf.joblib")

# Đánh giá
for model, name in zip([svm, rf], ["SVM", "Random Forest"]):
    print(f"\nĐánh giá mô hình {name}:")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))