import numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import os
import splitfolders
from plotter import plot

BASE_DIR = os.path.dirname(__file__)
DATASET_FOLDER = "output-gray-all-5"
DATASET_FOLDER_SPLIT = DATASET_FOLDER + "-split"
DATASET_DIR = BASE_DIR + "\\" + DATASET_FOLDER_SPLIT
MODELS_DIR = BASE_DIR + "\\models" + " (" + DATASET_FOLDER + ")"

size = 454
Learning_Rate = 1e-3
Epochs = 8
Batch_Size = 16
pooling_size =(3, 3)

splitfolders.ratio(DATASET_FOLDER, output=DATASET_FOLDER_SPLIT, seed=123406789, ratio=(.7, .2, .1), group_prefix=None)

train_dataset = ImageDataGenerator(rescale=1./255).flow_from_directory(
													batch_size=Batch_Size,
													directory=DATASET_DIR+"\\train",
													shuffle=True,
													target_size=(size, size),
													class_mode='categorical',
													color_mode='rgb',
													interpolation='lanczos')

validation_dataset = ImageDataGenerator(rescale=1./255).flow_from_directory(
														batch_size=Batch_Size,
														directory=DATASET_DIR+"\\val",
														shuffle=True,
														target_size=(size, size),
														class_mode='categorical',
														color_mode='rgb',
														interpolation='lanczos')

test_dataset = ImageDataGenerator(rescale=1./255).flow_from_directory(
														batch_size=Batch_Size,
														directory=DATASET_DIR+"\\test",
														shuffle=True,
														target_size=(size, size),
														class_mode='categorical',
														color_mode='rgb',
														interpolation='lanczos')

groups = train_dataset.num_classes

VGGBody = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(size, size, 3)))

VGGHead = VGGBody.output
VGGHead = AveragePooling2D(pool_size=pooling_size)(VGGHead)
VGGHead = Flatten(name="flatten")(VGGHead)
VGGHead = Dense(128, activation="relu")(VGGHead)
VGGHead = Dropout(0.5)(VGGHead)
VGGHead = Dense(groups, activation="softmax")(VGGHead)
VGGModel = Model(inputs=VGGBody.input, outputs=VGGHead)

for layer in VGGBody.layers: layer.trainable = False

ResNetBody = ResNet152V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(size, size, 3)))

ResNetHead = ResNetBody.output
ResNetHead = AveragePooling2D(pool_size=pooling_size)(ResNetHead)
ResNetHead = Flatten(name="flatten")(ResNetHead)
ResNetHead = Dense(128, activation="relu")(ResNetHead)
ResNetHead = Dropout(0.5)(ResNetHead)
ResNetHead = Dense(groups, activation="softmax")(ResNetHead)

ResNet = Model(inputs=ResNetBody.input, outputs=ResNetHead)

for layer in ResNetBody.layers: layer.trainable = False

InceptionBody = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(size, size, 3)))

InceptionHead = InceptionBody.output
InceptionHead = AveragePooling2D(pool_size=pooling_size)(InceptionHead)
InceptionHead = Flatten(name="flatten")(InceptionHead)
InceptionHead = Dense(128, activation="relu")(InceptionHead)
InceptionHead = Dropout(0.5)(InceptionHead)
InceptionHead = Dense(groups, activation="softmax")(InceptionHead)

InceptionModel = Model(inputs=InceptionBody.input, outputs=InceptionHead)

for layer in InceptionBody.layers: layer.trainable = False

opt = Adam(learning_rate=Learning_Rate, decay=Learning_Rate / Epochs)

try:
    os.mkdir(MODELS_DIR)
except FileExistsError:
    pass

ResNet.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
R = ResNet.fit(train_dataset,
				steps_per_epoch=len(train_dataset.filenames) / Batch_Size,
				validation_data=validation_dataset,
				validation_steps=len(validation_dataset.filenames) / Batch_Size,
				epochs=Epochs)
plot(R,"ResNet152V2",MODELS_DIR)

ResNet.save(MODELS_DIR + '\\ResNet_model.h5')

VGGModel.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
V = VGGModel.fit(train_dataset,
				steps_per_epoch=len(train_dataset.filenames) / Batch_Size,
				validation_data=validation_dataset,
				validation_steps=len(validation_dataset.filenames) / Batch_Size,
				epochs=Epochs)
plot(V,"VGG16",MODELS_DIR)
VGGModel.save(MODELS_DIR+'\\VGG_model.h5')

InceptionModel.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
I = InceptionModel.fit(train_dataset,
						steps_per_epoch=len(train_dataset.filenames) / Batch_Size,
						validation_data=validation_dataset,
						validation_steps=len(validation_dataset.filenames) / Batch_Size,
						epochs=Epochs)
plot(I,"InceptionV3",MODELS_DIR)
InceptionModel.save(MODELS_DIR+'\\InceptionV3_model.h5')

result_txt = open(MODELS_DIR + "\\Results.txt", 'w')
print("ResNet Results")
result_txt.write("ResNet Results: " + ResNet.evaluate(test_dataset).__str__() + "\n")
print("")

print("VGG16 Results")
result_txt.write("VGG16 Results: " + VGGModel.evaluate(test_dataset).__str__()+"\n")
print("")

print("Inception_V3 Results")
result_txt.write("Inception_V3 Results: " + InceptionModel.evaluate(test_dataset).__str__()+"\n")
result_txt.close()
