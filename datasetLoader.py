import csv
import os

class dl:
    BASE_DIR = os.path.dirname(__file__)
    datasetcsv = None
    datasetDir = {}
    datasetImg = {}

    def __init__(self,datasetcsv):
        self.datasetcsv = datasetcsv

        datasetcsv = self.datasetcsv
        datasetDir = self.datasetDir
        datasetImg = self.datasetImg
        BASE_DIR = self.BASE_DIR

        with open(datasetcsv, mode='r') as file:
            csvFile = csv.reader(file)

            for lines in csvFile:
                if lines[0] == "Category":
                    continue
                datasetDir[lines[0]] = lines[1]

            for group in datasetDir:
                groupPath = BASE_DIR + "\\" + datasetDir[group]
                onlyfiles = [f for f in os.listdir(groupPath) if os.path.isfile(os.path.join(groupPath, f))]

                fileDir = []
                for files in onlyfiles:
                    fileDir.append(groupPath + "\\" + files)
                datasetImg[group] = fileDir

    def getDatasetImg_dic(self):
        return self.datasetImg

    def getDatasetDir_dic(self):
        return self.datasetDir