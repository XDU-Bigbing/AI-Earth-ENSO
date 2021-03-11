import numpy as np

for i in range(10):
    a = np.random.rand(12, 24, 72, 4)
    np.save("tcdata/enso_round1_test_20210201/{}.npy".format(i), a)
