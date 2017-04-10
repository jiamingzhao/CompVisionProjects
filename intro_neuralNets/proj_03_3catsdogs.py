# Jiaming Zhao

# This fine-tuning code was taken from Keras documentation and modified slightly to have only

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input
import time

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'cats_and_dogs_medium/train'      # Path to training images
validation_data_dir = 'cats_and_dogs_medium/test'  # Validation and test set are the same here
nb_train_samples = 30000
nb_validation_samples = 900
epochs = 2
batch_size = 16
NUM_LAYERS = 19

try:
    model = load_model('cats_dogs_model.h5')
    print('Using cats_dogs_model.h5 model from directory')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

except:
    print('cats_dogs_model.h5 has not been found in directory. Training new model...')
    # Build the VGG16 network
    input_tensor = Input(shape=(150,150,3))
    base_model = VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
    # Add an additional MLP model at the "top" (end) of the network
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(1, activation='sigmoid'))
    model = Model(input= base_model.input, output= top_model(base_model.output))
    model.summary()

    # Freeze all the layers in the original model (fine-tune only the added Dense layers)
    for layer in model.layers[:NUM_LAYERS]:
        layer.trainable = False

    # Compile the model with a SGD/momentum optimizer and a slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])

    # Prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    start_time = time.time()
    # Fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples//batch_size,
        nb_epoch=epochs,				            # For Keras 2.0 API change to epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples//batch_size)       # For Keras 2.0 API change to validation_steps=nb_validation_samples
    print('Training time:', time.time() - start_time)
    model.save('cats_dogs_model.h5')
    print('model has been saved as cats_dogs_model.h5')


score = model.evaluate_generator(validation_generator, steps=nb_validation_samples)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
