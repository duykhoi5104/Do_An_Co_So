import tensorflow as tf
import os


train_dir = "/Users/ttdat/Documents/Do_An_Co_So/code/code_final/data_split/train"
test_dir = "/Users/ttdat/Documents/Do_An_Co_So/code/code_final/data_split/test"

img_size = (160, 160)
batch_size = 32

# Load ảnh thành dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'  # vì softmax
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Tối ưu hiệu suất
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Mô hình MobileNetV2 + phân lớp
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(train_ds.element_spec[1].shape[-1], activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Lưu model
model.save("model_mobilenetv2.h5")
print("\n[✓] Đã lưu model MobileNetV2 vào model_mobilenetv2.h5")