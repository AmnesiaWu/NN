import numpy as np

ans = np.arange(24).reshape((2, 3, 4))
np.save('d.npy', ans)
