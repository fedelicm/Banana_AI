from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import datasetLoader
from PIL import Image
import os
import csv

BASE_DIR = os.path.dirname(__file__)

dl = datasetLoader.dl("dataset.csv")
OUTPUT_FOLDER = "\\output"
OUTPUT_DIR = BASE_DIR + OUTPUT_FOLDER
datasetImg = dl.getDatasetImg_dic()
datasetDir = dl.getDatasetDir_dic()

try:
    os.mkdir(OUTPUT_DIR)
except FileExistsError:
    pass

with open('aug_dataset.csv', 'w', newline='') as csvfile:
    aug_csv_writer = csv.writer(csvfile, delimiter=',',)
    aug_csv_writer.writerow(["Category","Directory"])

for group in datasetImg:
    imgcount = 1

    output_group_dir = OUTPUT_DIR + "\\" + group
    try:
        os.mkdir(output_group_dir)
    except FileExistsError:
        pass

    with open('aug_dataset.csv', 'a', newline='') as csvfile:
        aug_csv_writer = csv.writer(csvfile, delimiter=',',)
        aug_csv_writer.writerow([group, OUTPUT_FOLDER + "\\" + group])

    for imgDir in datasetImg[group]:
        img = load_img(imgDir)

        data = img_to_array(img)

        samples = expand_dims(data, 0)

        datagen = ImageDataGenerator(featurewise_center=True, brightness_range=[0.5, 1.5],
                                     zoom_range=0.2, channel_shift_range=0.2,
                                     horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
        it = datagen.flow(samples, batch_size=1)

        for i in range(6):
            batch = it.next()
            image = batch[0].astype('uint8')
            im = Image.fromarray(batch[0].astype('uint8'))

            zeroes = ""
            for c in range(3-len(imgcount.__str__())):
                zeroes += "0"
            final_dir = output_group_dir+"\\"+zeroes+imgcount.__str__()+".jpg"
            im.save(final_dir)
            imgcount += 1


print("Image Augmentation Complete")
print("Images saved at " + OUTPUT_DIR)