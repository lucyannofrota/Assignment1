from tools import load_dataset as ldts
from tools import feature_name_gen as fng
import tools.feature_extraction as fe
import pandas as pd
import numpy as np

dts_path = '../dts/'
dataset_path = 'dataset/images/'
test = True

def create_dataset(test):
    imgs, labels = ldts.load_dataset(dataset_path, test=test)

    df = pd.DataFrame()

    S_images = []
    for img in imgs:
        S_images.append(fe.feature_SerializedImg(img=img))

    S_images = np.array(S_images)
    print("Data shape: ",S_images.shape)

    df = pd.DataFrame(S_images) # Images
    df.columns = fng.gen_name("img", 784)

    # ColorHistogram = []
    # for img in imgs:
    #     ColorHistogram.append(fe.feature_colorHistogram(img=img))

    # ColorHistogram = np.array(ColorHistogram)
    # ColorHistogram = np.squeeze(ColorHistogram)
    # print("Data shape: ", ColorHistogram.shape)

    # temp_df = pd.DataFrame(ColorHistogram)
    # temp_df.columns = fng.gen_name("CHistogram", 256)
    # df = pd.concat([df, temp_df], axis=1)
    # print("Dataframe shape: ", df.shape)


    gBlur = []
    for img in imgs:
        gb = fe.feature_gBlur(img=img)
        gBlur.append(fe.feature_SerializedImg(img=gb))

    gBlur = np.array(gBlur)
    print("Data shape: ", gBlur.shape)

    temp_df = pd.DataFrame(gBlur)
    temp_df.columns = fng.gen_name("gBlur", 784)
    df = pd.concat([df, temp_df], axis=1)
    print("Dataframe shape: ", df.shape)

    cMean = []
    for img in imgs:
        cMean.append(fe.feature_colorMean(img=img))

    cMean = np.array(cMean)
    print("Data shape: ", cMean.shape)

    temp_df = pd.DataFrame(cMean)
    temp_df.columns = ["MeanColor"]
    df = pd.concat([df, temp_df], axis=1)
    print("Dataframe shape: ", df.shape)


    # cVar = []
    # for img in imgs:
    #     cVar.append(fe.feature_var(img=img))

    # cVar = np.array(cVar)
    # print("Data shape: ", cVar.shape)

    # temp_df = pd.DataFrame(cVar)
    # temp_df.columns = fng.gen_name("Variance", 28)
    # df = pd.concat([df, temp_df], axis=1)
    # print("Dataframe shape: ", df.shape)



    temp_df = pd.DataFrame(labels)
    temp_df.columns = ["Label"]
    df = pd.concat([df, temp_df], axis=1)
    if test:
        df.to_csv(dts_path + 'test_3.csv',index=False)
    else:
        df.to_csv(dts_path + 'train_3.csv',index=False)

create_dataset(True)
create_dataset(False)