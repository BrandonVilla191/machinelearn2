from PIL import Image
from skimage.feature import hog
import numpy as np
from sklearn.decomposition import PCA

import os
import pickle

processed_images_data = None
processed_hogs_data = None
processed_pca_images_data = None
processed_pca_hogs_data = None
labels = None
for i, category in enumerate(["Corn___Healthy", "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Northern_Leaf_Blight"]):
  file_names = os.listdir(f"./Crop Diseases Dataset/{category}")
  print(len(file_names))
  if labels is not None:
    labels = np.concatenate((labels, np.full(len(file_names), i)))
  else:
    labels = np.full(len(file_names), i)
    
  for file_name in file_names:
    image = Image.open(f"Crop Diseases Dataset/{category}/{file_name}")
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(4, 4), visualize=True, channel_axis=-1)
    np_image = np.array(image)
    if processed_images_data is not None:
      processed_images_data = np.concatenate((processed_images_data, np_image.flatten()[np.newaxis]))
      processed_hogs_data = np.concatenate((processed_hogs_data, fd[np.newaxis]))
    else:
      processed_images_data = np_image.flatten()[np.newaxis]
      processed_hogs_data = fd[np.newaxis]
    print(processed_images_data.shape, processed_hogs_data.shape)
    
processed_images_with_labels = np.concatenate((processed_images_data, labels[:,np.newaxis]), axis=1)
processed_hogs_with_labels = np.concatenate((processed_hogs_data, labels[:,np.newaxis]), axis=1)
print(processed_images_with_labels.shape, processed_hogs_with_labels.shape)

with open(f"processed_images_with_labels.pickle", "wb") as f:
  pickle.dump(np.array(processed_images_with_labels), f)
with open(f"processed_hogs_with_labels.pickle", "wb") as f:
  pickle.dump(np.array(processed_hogs_with_labels), f)
  
image_pca = PCA(n_components=min(len(processed_images_data), 400))
hog_pca = PCA(n_components=min(len(processed_hogs_data), 100))
  
pca_processed_images = image_pca.fit_transform(processed_images_data)
pca_processed_hogs = hog_pca.fit_transform(processed_hogs_data)
print(pca_processed_images.shape, pca_processed_hogs.shape)

pca_processed_images_with_labels = np.concatenate((pca_processed_images, labels[:,np.newaxis]), axis=1)
pca_processed_hogs_with_labels = np.concatenate((pca_processed_hogs, labels[:,np.newaxis]), axis=1)
print(pca_processed_images_with_labels.shape, pca_processed_hogs_with_labels.shape)

with open(f"pca_processed_images_with_labels.pickle", "wb") as f:
  pickle.dump(np.array(pca_processed_images_with_labels), f)
with open(f"pca_processed_hogs_with_labels.pickle", "wb") as f:
  pickle.dump(np.array(pca_processed_hogs_with_labels), f)