import numpy as np

data = np.load('../dataset/KnowAir.npy')

print(data.shape)
print(data[0, 0, :])
