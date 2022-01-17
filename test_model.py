import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

BASE_DIR = os.path.dirname(__file__)
DATASET_FOLDER = "\\output-gray-all-5-split-2"
DATASET_DIR = BASE_DIR + DATASET_FOLDER
MODELS_DIR = BASE_DIR + "\\models-gray-all-5"

size = 454
Learning_Rate = 1e-3
Batch_Size = 16

validation_dataset = ImageDataGenerator(rescale=1./255).flow_from_directory(
														batch_size=Batch_Size,
														directory=DATASET_DIR+"\\test",
														shuffle=True,
														target_size=(size, size),
														class_mode='categorical',
														color_mode='rgb',
														interpolation='lanczos')

VGGModel = tf.keras.models.load_model(MODELS_DIR+'\\VGG_model.h5')
ResNet = tf.keras.models.load_model(MODELS_DIR+'\\ResNet_model.h5')
InceptionModel = tf.keras.models.load_model(MODELS_DIR+'\\InceptionV3_model.h5')

print("ResNet Results")
ResNet.evaluate(validation_dataset)
print("")

print("VGG16 Results")
VGGModel.evaluate(validation_dataset)
print("")

print("Inception_V3 Results")
InceptionModel.evaluate(validation_dataset)

