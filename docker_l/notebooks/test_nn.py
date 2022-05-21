from lib.data import load_dataset

x_train, y_train, x_test, y_test = load_dataset('dts/dts_plain/train.csv')

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(x_train)

x_train = sc.transform(x_train.values)

from nn_lucy import dts

train_dts = dts(x_train,y_train)

from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

training_loader = DataLoader(train_dts, batch_size=32, shuffle=True)
# testing_loader=DataLoader(testing_set, batch_size=8, shuffle=True)

from nn_lucy import nnModel


model = nnModel(inFeatures = x_train.shape[1], random_state=2)
# train(number_of_epochs=10, net=model, training_loader=training_loader)
losses, accs = model.fit(
    trainloader=training_loader,
    learningRate=0.01,
    momentum=0.9,
    gamma=1,
    numEpochs=21,
    verbose=False
)

print(losses)

import matplotlib as plt

plt.plot(losses)