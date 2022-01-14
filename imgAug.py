from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import datasetLoader
from PIL import Image
import os

BASE_DIR = os.path.dirname(__file__)

dl = datasetLoader.dl("dataset.csv")
OUTPUT_FOLDER = "output-gray-all-5"
OUTPUT_DIR = BASE_DIR + "\\" + OUTPUT_FOLDER
datasetImg = dl.getDatasetImg_dic()
datasetDir = dl.getDatasetDir_dic()
grayscale = True

try:
    os.mkdir(OUTPUT_DIR)
except FileExistsError:
    pass

for group in datasetImg:
    imgcount = 1

    output_group_dir = OUTPUT_DIR + "\\" + group
    try:
        os.mkdir(output_group_dir)
    except FileExistsError:
        pass

    for imgDir in datasetImg[group]:
        img = load_img(imgDir)

        data = img_to_array(img)

        samples = expand_dims(data, 0)

        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
        it = datagen.flow(samples, batch_size=1)

        for i in range(4):
            batch = it.next()
            im = Image.fromarray(batch[0].astype('uint8'))

            if(grayscale):
                im = im.convert('L')

            zeroes = ""
            for c in range(3-len(imgcount.__str__())):
                zeroes += "0"
            final_dir = output_group_dir+"\\"+zeroes+imgcount.__str__()+".jpg"
            im.save(final_dir)
            imgcount += 1


print("Image Augmentation Complete")
print("Images saved at " + OUTPUT_DIR)