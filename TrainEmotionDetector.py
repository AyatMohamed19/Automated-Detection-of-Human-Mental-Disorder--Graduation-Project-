from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Flatten, Activation
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from matplotlib import pyplot as plt

num_features = 32
num_labels = 7
batch_size = 64
epochs = 70
width, height = 48, 48

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1. / 255)
validation_data_gen = ImageDataGenerator(rescale=1. / 255)

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Build the model
model = Sequential()
model.add(Conv2D(2 * num_features, kernel_size=(3, 3), padding='same', data_format='channels_last',input_shape=(width, height, 1)))
model.add(Conv2D(2 * num_features, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(1, 1), padding='same'))
model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(1, 1), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=128))
model.add(BatchNormalization())
model.add(Dense(units=128))
model.add(BatchNormalization())

model.add(Dense(num_labels, activation='softmax'))

model.summary()

# Compiling the model with adam optimizer and categorical crossentropy loss
model.compile(loss=categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# Train the neural network/model
emotion_model_info = model.fit(
    train_generator,
    steps_per_epoch=32114 // 64,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=6423 // 64,
    # callbacks=[es]
)

# The below snippet plots the graph of the training loss vs. validation loss over the number of epochs.
# This will help the developer of the model to make informed decisions
# about the architectural choices that need to be made
# -----------------------------------------------------------------------
print(emotion_model_info.history.keys())

loss_train = emotion_model_info.history['loss']
loss_val = emotion_model_info.history['val_loss']
epochs = range(1, 71)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# ------------------------------------------------------------------------

# The following snippet plots the graph of training accuracy vs. validation accuracy over the number of epochs.
# ------------------------------------------------------------------------

loss_train = emotion_model_info.history['accuracy']
loss_val = emotion_model_info.history['val_accuracy']
epochs = range(1, 71)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# ------------------------------------------------------------------------


# saving the  model to be used later
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights("model.h5")
print("Saved model to disk")
