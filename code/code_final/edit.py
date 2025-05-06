from facenet_pytorch import MTCNN
from PIL import Image
import os

mtcnn = MTCNN(keep_all=False)

input_dir = "/Users/ttdat/Documents/Do_An_Co_So/dataset/Văn Tiến"
output_dir = os.path.join(input_dir, "/Users/ttdat/Documents/Do_An_Co_So/output/Văn Tiến" \
"")
os.makedirs(output_dir, exist_ok=True)

def expand_box_proportional(box, img_width, img_height, target_width=320, target_height=400):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # tâm mặt
    half_w, half_h = target_width / 2, target_height / 2
    nx1 = max(0, cx - half_w)
    ny1 = max(0, cy - half_h)
    nx2 = min(img_width, cx + half_w)
    ny2 = min(img_height, cy + half_h)
    return (nx1, ny1, nx2, ny2)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            box, prob = mtcnn.detect(img)

            if box is not None:
                bbox = expand_box_proportional(box[0], width, height, target_width=320, target_height=400)
                face_crop = img.crop(bbox).resize((320, 400))  # giữ đúng kích thước mẫu bạn gửi
                save_path = os.path.join(output_dir, f"crop_{filename}")
                face_crop.save(save_path)
                print(f"[✓] Crop + resize: {save_path}")
            else:
                print(f"[x] Không phát hiện mặt: {filename}")
        except Exception as e:
            print(f"[!] Lỗi với {filename}: {e}")