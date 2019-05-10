import datasets
import numpy as np 
import matplotlib.pyplot as plt

golf = datasets.GolfSwing()

x, y = golf[0]
y = np.array(y)

fig, ax = plt.subplots(1, 1)
ax.imshow(y, cmap='gray')
fig.savefig('temp.png')

import pdb; pdb.set_trace()