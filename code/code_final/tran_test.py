import os
import numpy as np
import random
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mô hình MTCNN + FaceNet
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Thư mục chứa ảnh của tất cả người
data_dir = "/Users/ttdat/Documents/Do_An_Co_So/dữ liệu chính"

# Danh sách vector và nhãn
X_train, y_train = [], []
X_test, y_test = [], []

# Duyệt từng người
for person in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue

    # Danh sách ảnh
    images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) < 3:
        print(f"[!] Bỏ qua {person} vì quá ít ảnh ({len(images)})")
        continue

    random.shuffle(images)
    n_train = int(len(images) * 0.7)
    train_imgs = images[:n_train]
    test_imgs = images[n_train:]

    def process(img_list, X, y, tag):
        for img_name in img_list:
            try:
                img_path = os.path.join(person_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                face = mtcnn(img)
                if face is not None:
                    face = face.unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model(face).cpu().numpy()[0]
                    X.append(emb)
                    y.append(person)
                    print(f"[✓] {tag}: {person}/{img_name}")
                else:
                    print(f"[x] Không nhận diện được: {person}/{img_name}")
            except Exception as e:
                print(f"[!] Lỗi ảnh {person}/{img_name}: {e}")

    process(train_imgs, X_train, y_train, "Train")
    process(test_imgs, X_test, y_test, "Test")

# Lưu kết quả
np.savez("train_embeddings.npz", X=np.array(X_train), y=np.array(y_train))
np.savez("test_embeddings.npz", X=np.array(X_test), y=np.array(y_test))

print(f"\n[✓] HOÀN TẤT — Train: {len(X_train)} ảnh, Test: {len(X_test)} ảnh")