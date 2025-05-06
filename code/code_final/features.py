from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Đường dẫn đến các ảnh đã upload
image_dir = "/Users/ttdat/Documents/Do_An_Co_So/dữ liệu chính/Văn Tiến"  
image_files = [
    "vt1.jpeg", "vt2.jpeg", "vt3.jpeg",  
    "vt4.jpeg", "vt5.jpeg", "vt6.jpeg"
]

embeddings = []
labels = []

for filename in image_files:
    img_path = os.path.join(image_dir, filename)
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = facenet(face).cpu().numpy()
        embeddings.append(emb[0])
        labels.append("Van Tien")
    else:
        print(f"[x] Không nhận diện được khuôn mặt: {filename}")

# Lưu kết quả
np.savez("face_embeddings_VanTien.npz", features=np.array(embeddings), labels=np.array(labels))
print(f"[✓] Đã trích đặc trưng xong cho {len(embeddings)} ảnh của Văn Tiến.")