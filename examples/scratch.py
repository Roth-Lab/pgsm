import numpy as np
data = np.random.normal(0, 1, size=100)
clustering = np.random.randint(0, 4, size=data.shape[0])
anchor_idx = np.random.choice(np.arange(data.shape[0]), replace=False, size=2)
anchor_idx
sigma = np.argwhere(
    (clustering == clustering[anchor_idx[0]]) | (clustering == clustering[anchor_idx[1]])
).flatten()
sigma = list(sigma)
[sigma.remove(x) for x in anchor_idx]
np.random.shuffle(sigma)
sigma
