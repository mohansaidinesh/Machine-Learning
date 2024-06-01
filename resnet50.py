import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the ResNet model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
base_model.trainable = False  # Freeze the ResNet layers

model = Sequential()

# Add the ResNet base model
model.add(base_model)

# Flatten the output of the ResNet model
model.add(Flatten())

# Add custom dense layers
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    '/content/drive/My Drive/data/train',
    target_size=(100, 100),
    batch_size=64,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    '/content/drive/My Drive/data/val',
    target_size=(100, 100),
    batch_size=64,
    class_mode='binary'
)

my_callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('my_model_resnet.h5', verbose=1, save_best_only=True, save_weights_only=False)
]

model.fit(training_set, epochs=200, validation_data=test_set, callbacks=my_callbacks)

model.save('my_model_resnet.h5')
