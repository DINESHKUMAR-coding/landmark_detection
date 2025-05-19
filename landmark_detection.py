import numpy as np
import pandas as pd
import os
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Data
df = pd.read_csv("train.csv")
df = df[df['id'].str.startswith('00')]  # Correct filter
num_classes = len(df["landmark_id"].unique())

# Label encoding
lencoder = LabelEncoder()
df["label_encoded"] = lencoder.fit_transform(df["landmark_id"])

# Display class distribution
data = pd.DataFrame(df["landmark_id"].value_counts())
data.reset_index(inplace=True)
data.columns = ['landmark_id', 'count']
plt.hist(data['count'], bins=100)
plt.title('Class distribution')
plt.show()

# Image reading helper
base_path = "./0"

def get_image_from_number(idx, dataframe):
    fname = dataframe.iloc[idx]['id'] + '.jpg'
    label = dataframe.iloc[idx]['label_encoded']
    path = os.path.join(base_path, fname[0], fname[1], fname[2], fname)
    im = cv2.imread(path)
    return im, label

# Show random images
fig = plt.figure(figsize=(16,16))
for i in range(1,5):
    rand_idx = random.randint(0, len(df) - 1)
    img, label = get_image_from_number(rand_idx, df)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig.add_subplot(1, 4, i)
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    return img

# Model
base_model = VGG19(weights=None, include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
optimizer = RMSprop(learning_rate=0.0001, decay=1e-6, momentum=0.9)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train-Validation Split
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

# Data Generator
def data_generator(dataframe, batch_size):
    while True:
        batch_indices = np.random.choice(len(dataframe), batch_size)
        image_batch = []
        label_batch = []
        for idx in batch_indices:
            img, label = get_image_from_number(idx, dataframe)
            if img is not None:
                img = preprocess_image(img)
                image_batch.append(img)
                label_batch.append(label)
        yield np.array(image_batch), np.array(label_batch)

batch_size = 16
epochs = 10

# Training
steps_per_epoch = len(train_df) // batch_size
validation_steps = len(val_df) // batch_size

model.fit(
    data_generator(train_df, batch_size),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(val_df, batch_size),
    validation_steps=validation_steps,
    epochs=epochs
)

# Save model
model.save("landmark_detector.h5")
print("Model saved successfully!")


