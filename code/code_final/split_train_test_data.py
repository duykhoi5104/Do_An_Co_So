import os
import numpy as np
import glob
import shutil

folder_path = 'C:/Users/khoi0/Downloads/Git/Do_An_Co_So/code/saved_embeddings'

file_names = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
print(file_names)

# Tổng số lượng file
total_files = len(file_names)
print(f"Tổng số file: {total_files}")

# Tính chỉ số chia 70:30
split_index = int(0.7 * total_files)
train_files = file_names[:split_index]
test_files = file_names[split_index:]

# Kiểm tra số lượng file sau khi chia
print(len(train_files))
print(len(test_files))

# In danh sách file train
print(train_files)
# In danh sách file test
print(test_files)

# Tạo thư mục mới nếu chưa tồn tại
# Thiết lập đường dẫn thư mục train_set và test_set
path_train_test_data = 'C:/Users/khoi0/Downloads/Git/Do_An_Co_So/code'
train_folder = os.path.join(path_train_test_data, 'train_set')
test_folder = os.path.join(path_train_test_data, 'test_set')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Sao chép các file train vào thư mục train_files
for file_name in train_files:
    src_path = os.path.join(folder_path, file_name)
    dst_path = os.path.join(train_folder, file_name)
    shutil.copy2(src_path, dst_path)

# Sao chép các file test vào thư mục test_files
for file_name in test_files:
    src_path = os.path.join(folder_path, file_name)
    dst_path = os.path.join(test_folder, file_name)
    shutil.copy2(src_path, dst_path)

print(f"Đã sao chép {len(train_files)} file vào thư mục: {train_folder}")
print(f"Đã sao chép {len(test_files)} file vào thư mục: {test_folder}")