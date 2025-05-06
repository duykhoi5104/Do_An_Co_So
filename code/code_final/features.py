from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import numpy as np
import os
from imagetools import histogram_equalization, get_image_list, save_image

# Khởi tạo 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Lấy dữ liệu 
data_root = "C:/Users/Loc/Desktop/DACS_git/Do_An_Co_So/dữ liệu chính"

# Thư mục lưu đặc trưng 
save_root = "C:/Users/Loc/Desktop/DACS_git/Do_An_Co_So/code/saved_embeddings"
os.makedirs(save_root, exist_ok=True)

# Duyệt qua từng thư mục (nhãn được đặt tên theo người đó)
for person_name in os.listdir(data_root):
    person_folder = os.path.join(data_root, person_name)
    if not os.path.isdir(person_folder):
        continue

    embeddings = []
    labels = []

    print(f"Đang xử lý: {person_name}")
    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_folder, filename)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert("RGB")
            
            face = mtcnn(img)

            if face is not None:
                face = face.unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = facenet(face).cpu().numpy()
                embeddings.append(emb[0])
                labels.append(person_name)
            else:
                print(f"Không nhận diện được khuôn mặt: {filename} ({person_name})")

    # Lưu đặc trưng của từng người vào file 
    if embeddings:
        save_path = os.path.join(save_root, f"{person_name}.npz")
        np.savez(save_path, features=np.array(embeddings), labels=np.array(labels))
        print(f"Đã lưu đặc trưng của {person_name} vào {save_path}")
    else:
        print(f"Không có đặc trưng nào được trích từ {person_name}")
