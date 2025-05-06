import numpy as np

data = np.load("face_embeddings_Khoi.npz")
features = data["features"]
labels = data["labels"]

print("Số ảnh:", features.shape[0])
print("Chiều dài mỗi vector:", features.shape[1])
print("Nhãn:", np.unique(labels))
print("Vector đầu tiên (5 giá trị đầu):", features[0][:5])