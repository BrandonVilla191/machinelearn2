# 7641-project 

These models aims to quickly and accurately identify diseases in crops like corn using machine learning. This issue is critical because crop diseases can lead to major losses in yield and quality, significantly impacting the economy. Our goal is to create a tool to help farmers easily detect these diseases, leading to better decision-making, healthier crops, and increased productivity.

## Kmeans

##### Kmeans\clustering_eval.py
Metrics used to test the clustering of the kmeans model.There is a silhouette_coefficient function to calculate the silhouette score, which measures similarity within a cluster compared to between clusters, and a beta_cv function to evaluate the clustering. Beta_cv measures the ratio of the average between intra-cluster distance using pairwise to the average nearest cluster distance. It measures compactness and separation. With these, we measure the efficacy of the model.

##### Kmeans\kmeans.ipynb
The model is run in this juypter notebook file. The pca and hog .pickle dataset is used. The featureset of both .pickle files is combined then PCA is applied to this to reduce its dimensionality to 100. Data is split into training and testing, and the clustering is set to 4 to correspond to distinct image features and with a maximum of 1000 iterations for convergence. After fitting and training the data, it is applied to test_features to test the clustering on unseen data and the results are saved into a .pickle file. Evaluation metrics and visualizations (beta_cv, confusion matrix, Silhouette Coefficient, Accuracy, and Calinski-Harabasz index) are then provided to view the efficacy of the clustering.

##### Kmeans\kmeans.pdf
This is a pdf representation of the model in the notebook

##### Kmeans\preprocess.py
Prepocessing for the model implementation. Data is gathered from four different categories: healthy corn, common rust in corn, corn with gray leaf spot, and corn with northern leaf blight. Here we load the images for each category and create HOG descriptors for each image, with captures texture and shape features that our kmeans model can use to differentiate between different corn conditions. These are then saved to .pickle files. PCA is then applied to the both the raw image pixels and HOG descriptors to reduce the dimensionality of the data. Afterwards, the reduced data is again saved in .pickle files, which contain the feature sets and can be used by our model.

### Kmeans\processed-data
This directory contains the input data that the kmeans model used, as well as the trained model. `pca_processed_hogs_with_labels.pickle` and `pca_processed_images_with_lables.pickle` contain HOG descriptors and raw image files that have been reduced with PCA and paired with labels for input into the model. `kmeans_model_images.pickle` and `kmeans_model_hogs.pickle` stores the trained k-means clustering model.

