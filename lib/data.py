dataset_link = "https://www.dropbox.com/s/uje00mjhmki15ze/datasets.rar?dl=0"

class Dataset:
    def __init__(self, folder_name):
        self.data = self.__import__(folder_name)
        self.__categorize__()
    
    def __import__(self):
        
        pass
    def __categorize__(self):
        pass

class Data:
    def __init__(self, image, label):
        self.image = image
        self.label = label