from facenet_pytorch import MTCNN
from PIL import Image
import os

# Khởi tạo MTCNN
mtcnn = MTCNN(image_size=160, margin=20)

# Thư mục gốc chứa ảnh theo người
input_dir = '/Users/ttdat/Documents/Do_An_Co_So/dataset/khôi'               # ← bạn đổi tên đúng theo thư mục bạn có
output_dir = '/Users/ttdat/Documents/Do_An_Co_So/output/khôi'        # ← nơi sẽ lưu ảnh sau khi cắt

os.makedirs(output_dir, exist_ok=True)

# Duyệt qua từng người
for person_name in os.listdir(input_dir):
    person_folder = os.path.join(input_dir, person_name)

    if not os.path.isdir(person_folder):
        continue  # bỏ qua nếu không phải thư mục

    save_folder = os.path.join(output_dir, person_name)
    os.makedirs(save_folder, exist_ok=True)

    for image_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image_name)

        try:
            img = Image.open(img_path)
            save_path = os.path.join(save_folder, image_name)
            mtcnn(img, save_path=save_path)
            print(f"Đã cắt: {image_name} → {person_name}")
        except Exception as e:
            print(f"Lỗi ảnh {image_name}: {e}")