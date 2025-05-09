import os
import shutil
import random

source_dir = "/Users/ttdat/Documents/Do_An_Co_So/dữ liệu chính"  
output_dir = "data_split"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for person in os.listdir(source_dir):
    person_path = os.path.join(source_dir, person)
    if not os.path.isdir(person_path):
        continue

    images = [f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(images) < 3:
        print(f"[!] Bỏ qua {person} vì quá ít ảnh ({len(images)})")
        continue

    random.shuffle(images)
    n_train = int(len(images) * 0.7)
    train_imgs = images[:n_train]
    test_imgs = images[n_train:]

    os.makedirs(os.path.join(train_dir, person), exist_ok=True)
    os.makedirs(os.path.join(test_dir, person), exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(person_path, img), os.path.join(train_dir, person, img))

    for img in test_imgs:
        shutil.copy2(os.path.join(person_path, img), os.path.join(test_dir, person, img))

    print(f" {person}: Train={len(train_imgs)} | Test={len(test_imgs)}")

print("\n Đã chia xong ảnh vào data_split/train và data_split/test")