import os
from PIL import Image
import numpy as np

# Đọc ảnh từ đường dẫn
def load_image(image_path):
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print("Lỗi khi đọc hình ảnh từ: ", image_path, " ", e)
        return None

# Kiểm tra xem có phải là ảnh không
def is_image_file(file_path):
    extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    return file_path.lower().endswith(extensions)

# Cân bằng histogram của ảnh
def histogram_equalization(image, nbr_bins=256):
    if image.mode != 'L':
        image = image.convert('L')  # Đảm bảo ảnh là ảnh xám
    image_array = np.array(image)
    histogram, bins = np.histogram(image_array, bins=nbr_bins, range=(0, 256), density=True)
    cdf = histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]
    image_equalized = np.interp(image_array, bins[:-1], cdf)
    return Image.fromarray(image_equalized.astype('uint8'))

# Lấy danh sách ảnh trong thư mục
def get_image_list(folder_path):
    image_list = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        filenames = os.listdir(folder_path)
        for filename in filenames:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and is_image_file(file_path):
                img = load_image(file_path)
                image_list.append(img)
    return image_list

# Lưu ảnh vào thư mục
def save_image(image, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image.save(os.path.join(folder_path, filename))
