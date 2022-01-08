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
output_dir = BASE_DIR + "\\output"
datasetImg = dl.getDatasetImg_dic()
datasetDir = dl.getDatasetDir_dic()

for group in datasetImg:
    imgcount = 1
    for imgDir in datasetImg[group]:
        img = load_img(imgDir)

        data = img_to_array(img)

        samples = expand_dims(data, 0)

        datagen = ImageDataGenerator(featurewise_center=True, brightness_range=[0.5, 1.5],
                                     zoom_range=0.2, channel_shift_range=0.2,
                                     horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
        it = datagen.flow(samples, batch_size=1)

        output_group_dir = output_dir + "\\" + group
        try:
            os.mkdir(output_group_dir)
        except FileExistsError:
            pass

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
print("Images saved at " + output_dir)