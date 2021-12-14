from sklearn.model_selection import train_test_split
import numpy as np

x = np.load('.\Data\CompleteHarryPotter3\PosX.npy')
y = np.load('.\Data\CompleteHarryPotter3\PosY.npy')
print(x.shape, y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.1)

np.save('.\Data\Complete566Data\PosX_train.npy', xTrain)
np.save('.\Data\Complete566Data\PosY_train.npy', yTrain)
np.save('.\Data\Complete566Data\PosX_test.npy', xTest)
np.save('.\Data\Complete566Data\PosY_test.npy', yTest)