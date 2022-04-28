from tools import load_dataset as ldts
from tools import feature_name_gen as fng
import tools.feature_extraction as fe
import pandas as pd
import numpy as np

dts_path = 'dataset/images/'
test = True
imgs, labels = ldts.load_dataset(dts_path, test=test)

df = pd.DataFrame()



# print(imgs[0])
S_images = []
for img in imgs:
    S_images.append(fe.feature_SerializedImg(img=img))

S_images = np.array(S_images)
print("Data shape: ",S_images.shape)

df = pd.DataFrame(S_images) # Images
df.columns = fng.gen_name("img", 784)

# gBlur = []
# for img in imgs:
#     gb = fe.feature_gBlur(img=img)
#     gBlur.append(fe.feature_SerializedImg(img=gb))

# gBlur = np.array(gBlur)
# print("Data shape: ", gBlur.shape)

# temp_df = pd.DataFrame(gBlur)
# temp_df.columns = fng.gen_name("gBlur", 784)
# df = pd.concat([df, temp_df], axis=1)
# print("Dataframe shape: ", df.shape)




temp_df = pd.DataFrame(labels)
temp_df.columns = ["Label"]
df = pd.concat([df, temp_df], axis=1)
if test:
    df.to_csv('test.csv',index=False)
else:
    df.to_csv('train.csv',index=False)