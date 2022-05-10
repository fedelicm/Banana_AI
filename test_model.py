import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy
import sklearn.metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
DATASET_FOLDER = "\\output-gray-all-5-split"
DATASET_DIR = BASE_DIR + DATASET_FOLDER
MODELS_DIR = BASE_DIR + "\\models-gray-all-5"
result_VGG = open(MODELS_DIR + "\\Results_VGG.txt", 'w')
result_ResNet = open(MODELS_DIR + "\\Results_ResNet.txt", 'w')
result_Inception = open(MODELS_DIR + "\\Results_Inception.txt", 'w')

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

true_classes = validation_dataset.classes

class_labels = list(validation_dataset.class_indices.keys())
true_classes_str = [class_labels[x] for x in true_classes]


predictions_ResNet = ResNet.predict_generator(validation_dataset, steps=Batch_Size)
predicted_classes_ResNet = numpy.argmax(predictions_ResNet, axis=1)
predicted_classes_ResNet_str = [class_labels[x] for x in predicted_classes_ResNet]
report_ResNet = confusion_matrix(true_classes_str, predicted_classes_ResNet_str, labels=class_labels)

disp_ResNet = ConfusionMatrixDisplay(confusion_matrix=report_ResNet, display_labels=class_labels)
disp_ResNet.plot()
report2 = sklearn.metrics.classification_report(true_classes, predicted_classes_ResNet, target_names=class_labels)
result_ResNet.write(report2)
result_ResNet.close()
plt.title("ResNet")
plt.savefig(MODELS_DIR + "\\" + "ResNet_ConfusionMatrix" + ".png")

predictions_VGGModel = VGGModel.predict_generator(validation_dataset, steps=Batch_Size)
predicted_classes_VGGModel = numpy.argmax(predictions_VGGModel, axis=1)
predicted_classes_VGGModel_str = [class_labels[x] for x in predicted_classes_VGGModel]
report_VGGModel = confusion_matrix(true_classes_str, predicted_classes_VGGModel_str, labels=class_labels)

disp_VGGModel = ConfusionMatrixDisplay(confusion_matrix=report_VGGModel, display_labels=class_labels)
disp_VGGModel.plot()
report2 = sklearn.metrics.classification_report(true_classes, predicted_classes_VGGModel, target_names=class_labels)
result_VGG.write(report2)
result_VGG.close()
plt.title("VGGModel")
plt.savefig(MODELS_DIR + "\\" + "VGGModel_ConfusionMatrix" + ".png")

predictions_InceptionModel = InceptionModel.predict_generator(validation_dataset, steps=Batch_Size)
predicted_classes_InceptionModel = numpy.argmax(predictions_InceptionModel, axis=1)
predicted_classes_InceptionModel_str = [class_labels[x] for x in predicted_classes_InceptionModel]
report_InceptionModel = confusion_matrix(true_classes_str, predicted_classes_InceptionModel_str, labels=class_labels)

disp_InceptionModel = ConfusionMatrixDisplay(confusion_matrix=report_InceptionModel, display_labels=class_labels)
disp_InceptionModel.plot()
report2 = sklearn.metrics.classification_report(true_classes, predicted_classes_InceptionModel,
												target_names=class_labels)
result_Inception.write(report2)
result_Inception.close()
plt.title("InceptionModel")
plt.savefig(MODELS_DIR + "\\" + "InceptionModel_ConfusionMatrix" + ".png")


print("ResNet Results")
ResNet.evaluate(validation_dataset)
print("")

print("VGG16 Results")
VGGModel.evaluate(validation_dataset)
print("")

print("Inception_V3 Results")
InceptionModel.evaluate(validation_dataset)

