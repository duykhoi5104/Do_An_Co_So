from PIL import Image
from imagetools import *
from IPython.display import display
#Thư mục chứa hình ảnh:
dir = "/Users/ttdat/Documents/Do_An_Co_So/dataset/khôi"

#Đọc nhiều ảnh:
imgs = get_image_list(dir)

#Hiển thị ảnh bằng hàm display:
for img in imgs:
    print(img.size)
    display(img)
    