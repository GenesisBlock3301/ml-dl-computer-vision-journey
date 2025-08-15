# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.models import Model
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# base_dir = './dataset/CamVid'
#
# def make_dataset(img_dir, label_dir):
#     img_files = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir))
#     label_files = sorted(os.path.join(label_dir, f) for f in os.listdir(label_dir))
#     ds = tf.data.Dataset.from_tensor_slices((img_files, label_files))
#     return ds
#
# train_ds = make_dataset(os.path.join(base_dir, 'train'), os.path.join(base_dir, 'train_labels'))
# val_ds = make_dataset(os.path.join(base_dir, 'val'), os.path.join(base_dir, 'val_labels'))
# test_ds = make_dataset(os.path.join(base_dir, 'test'), os.path.join(base_dir, 'test_labels'))
#
# print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
#
# def load_image(img_path, label_path):
#     _img = tf.io.read_file(img_path)
#     _img = tf.image.decode_png(_img, channels=3)
#     _img = tf.image.convert_image_dtype(_img, tf.float32)
#
#     _mask = tf.io.read_file(label_path)
#     _mask = tf.image.decode_png(_mask, channels=1)
#     return _img, _mask
#
#
# train_ds = train_ds.map(load_image)
# val_ds = val_ds.map(load_image)
# test_ds = test_ds.map(load_image)
#
#
#
# def unet_model(input_size=(256, 256, 3), num_classes=NUM_CLASSES):
#     inputs = layers.Input(input_size)
#
#     # Encoder
#     c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
#     p1 = layers.MaxPooling2D()(c1)
#
#     c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
#     c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
#     p2 = layers.MaxPooling2D()(c2)
#
#     # Bottleneck
#     b1 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
#     b1 = layers.Conv2D(256, 3, activation='relu', padding='same')(b1)
#
#     # Decoder
#     u1 = layers.UpSampling2D()(b1)
#     u1 = layers.Concatenate()([u1, c2])
#     c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
#     c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
#
#     u2 = layers.UpSampling2D()(c3)
#     u2 = layers.Concatenate()([u2, c1])
#     c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
#     c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)
#
#     outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c4)
#     return models.Model(inputs, outputs)
#
# model = unet_model()
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
#
# # ধরো তুমি ৫টা layer এর output দেখতে চাও
# layer_outputs = [layer.output for layer in model.layers[:10]]
# activation_model = Model(inputs=model.input, outputs=layer_outputs)
#
# # একটা sample image নাও (CamVid এর থেকে)
# img = np.random.rand(1, 256, 256, 3)  # ডেমোর জন্য random image
#
# # prediction করো
# activations = activation_model.predict(img)
#
# # এখন visualize করো
# layer_names = [layer.name for layer in model.layers[:10]]
# for layer_name, layer_activation in zip(layer_names, activations):
#     print(layer_name, layer_activation.shape)  # Feature map shape
#     n_features = layer_activation.shape[-1]  # কতগুলো channel
#     size = layer_activation.shape[1]
#     display_grid = np.zeros((size, size * n_features))
#     for i in range(n_features):
#         x = layer_activation[0, :, :, i]
#         x -= x.mean()
#         x /= (x.std() + 1e-5)
#         x *= 64
#         x += 128
#         x = np.clip(x, 0, 255).astype('uint8')
#         display_grid[:, i * size : (i + 1) * size] = x
#     plt.figure(figsize=(n_features, 1))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid, aspect='auto', cmap='viridis')


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread("../img/img2.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # optional, work with single channel

# Define a simple kernel (3x3 edge detection)
kernel = np.array([[ -1, -1, -1],
                   [ -1,  8, -1],
                   [ -1, -1, -1]], dtype=np.float32)

# Apply convolution
feature_map = cv2.filter2D(img, -1, kernel)

# Visualize
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Feature map")
plt.imshow(feature_map, cmap='gray')
plt.axis('off')
plt.show()
