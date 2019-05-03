import os
import sys
print(sys.path)
import json
import tensorflow as tf
import matplotlib.pyplot as plt 

sys.path.append(os.path.abspath('../'))

from dataset import MNIST
from policy import json2policy

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
policy_filename = "../models/policies/plot_subpolicy.json"
with open(policy_filename, "r") as f:
    policy_json = json.load(f)
policy = json2policy(policy_json)
    
session = tf.Session()
dataset = MNIST(batch_size = 3)
x, y = dataset.get_batch(data = "Train", policy = policy)

[b, h, w, c] = x.shape

fig, axis = plt.subplots(3, 6, figsize = (12, 4))

print(y.shape)
for i, ax in enumerate((axis.T).flatten()):
    ax.imshow(y[i].reshape([h, w]), cmap = 'gray')
    ax.axis('off')
    
axis[0, 0].set_title('Original')
axis[0, 0].text(-20, 14, "Image 1", horizontalalignment='center', fontsize = 12)
axis[1, 0].text(-20, 14, "Image 2", horizontalalignment='center', fontsize = 12)
axis[2, 0].text(-20, 14, "Image 3", horizontalalignment='center', fontsize = 12)
for i in range(1,6):
    axis[0, i].set_title('Subpolicy {}'.format(i))
    
fontsize = 10
axis[2, 1].text(15, 35, "ElasticDeform, 0.1, 1.4", horizontalalignment='center', fontsize = fontsize)
axis[2, 1].text(15, 40, "Shear, 0.5, 4.0", horizontalalignment='center', fontsize = fontsize)

axis[2, 2].text(15, 35, "FlipX, 0.3, -", horizontalalignment='center', fontsize = fontsize)
axis[2, 2].text(15, 40, "TranslateX, 0.3, -3.0", horizontalalignment='center', fontsize = fontsize)

axis[2, 3].text(15, 35, "FlipY, 0.3, -", horizontalalignment='center', fontsize = fontsize)
axis[2, 3].text(15, 40, "TranslateX, 0.8, -4.0", horizontalalignment='center', fontsize = fontsize)

axis[2, 4].text(15, 35, "TrasnlateY, 0.0, -1.0", horizontalalignment='center', fontsize = fontsize)
axis[2, 4].text(15, 40, "Shear, 0.5, 8.0", horizontalalignment='center', fontsize = fontsize)

axis[2, 5].text(15, 35, "FlipX, 0.5, -", horizontalalignment='center', fontsize = fontsize)
axis[2, 5].text(15, 40, "Elastic Deform, 0.7, 2.8", horizontalalignment='center', fontsize = fontsize)

plt.savefig('../PaperFigures/PolicySummary.eps')
plt.show()