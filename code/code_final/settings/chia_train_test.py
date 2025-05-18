import numpy as np
import os
import random

# Đường dẫn chứa các file .npz
data_root = "C:/Users/Loc/Desktop/Do_An_Co_So/Do_An_Co_So/code/saved_embeddings"

# Danh sách đặc trưng train và test
train_embeddings, train_labels = [], []
test_embeddings, test_labels = [], []

def determine_test_count(n):
    """Trả về số lượng test dựa trên tổng số ảnh của một người."""
    if n <= 6:
        return 1
    elif n <= 10:
        return 2
    else:
        return max(2, int(n * 0.25))  # ~25%

# Duyệt qua các file .npz
for file in os.listdir(data_root):
    if file.endswith(".npz"):
        path = os.path.join(data_root, file)
        data = np.load(path)
        features = data["features"]
        labels = data["labels"]
        n = len(features)

        # Shuffle ngẫu nhiên
        idx = list(range(n))
        random.shuffle(idx)

        num_test = determine_test_count(n)
        test_idx = idx[:num_test]
        train_idx = idx[num_test:]

        # Tách ảnh
        test_embeddings.append(features[test_idx])
        test_labels.append(labels[test_idx])
        train_embeddings.append(features[train_idx])
        train_labels.append(labels[train_idx])

# Gộp lại
X_train = np.vstack(train_embeddings)
y_train = np.concatenate(train_labels)
X_test = np.vstack(test_embeddings)
y_test = np.concatenate(test_labels)

# Lưu kết quả
save_dir = "C:/Users/Loc/Desktop/Do_An_Co_So/Do_An_Co_So/code/code_final/data_split"  
np.savez(os.path.join(save_dir, "train_embeddings.npz"), features=X_train, labels=y_train)
np.savez(os.path.join(save_dir, "test_embeddings.npz"), features=X_test, labels=y_test)

print(f" Đã chia dữ liệu ngẫu nhiên HỢP LÝ.")
print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")
print(f"Đã lưu tại: {save_dir}")
