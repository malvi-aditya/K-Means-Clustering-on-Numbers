#Import Required Libraries
from keras.datasets import mnist
import matplotlib.pyplot as plt,numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics


#Load Dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#Print Properties of dataset
print("x_train size: ",x_train.shape)
print("y_train size: ",y_train.shape)
print("x_test size: ",x_test.shape)
print("y_test size: ",y_test.shape)
print()


# Create figure with 3x3 subplots using matplotlib.pyplot
fig,axs=plt.subplots(3, 3, figsize = (12, 12))
plt.gray()
# loop through subplots and add mnist images
for i, ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.set_title('Number {}'.format(y_train[i]))
# Display the figure
print("Images: ")
fig.show()
plt.show()
print()


#Preprocess the Images
#Convert 28X28 Image to 1D array of length=784 (28*28)
X=x_train.reshape(len(x_train),-1)
Y=y_train
print("Reshaped x_train size: ",X.shape)
#normalize X data
X=X/255


#KMeans Algorithm

#No_digit=10 (0 to 9)
no_digit=len(np.unique(y_test))
print("No. of digit: ",no_digit)
#Initialize the Model
kmeans=MiniBatchKMeans(n_clusters=no_digit)
#fit the model
kmeans.fit(X)


def cluster_labels(kmeans,actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """
    infered_label={}
    for i in range(kmeans.n_clusters):
        # Find Index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)
        
        #Append actual labels for each point
        labels.append(actual_labels[index])
        
        #Determine the most common label
        if len(labels[0])==1:
            counts=np.bincount(labels[0])
        else:
            counts=np.bincount(np.squeeze(labels))
        #Assign cluster to a value in the dictionary
        if np.argmax(counts) in infered_label:
            # append the new number to the existing array at this slot
            infered_label[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            infered_label[np.argmax(counts)]=[i]
    return infered_label


def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    #Empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels


#Test the infer_cluster_labels() and infer_data_labels() functions
cluster_label=cluster_labels(kmeans, Y)
X_clusters = kmeans.predict(X)
predicted_labels = infer_data_labels(X_clusters, cluster_label)
print("First 20 predicted labels: ")
print(predicted_labels[:20])
print("Actual labels: ")
print(Y[:20])
print()
#optimizing and visualising


def calculate_metrics(estimator,data,label):
    
    #Calculate and print metrics
    print("No. of cluster: ",estimator.n_clusters)
    print("Inertia: ",estimator.inertia_)
    print("Homogeneity Score: ",metrics.homogeneity_score(label,estimator.labels_))
    
    
no_cluster=[10,16,36,64,144,256]
for i in no_cluster:
    #Fit the data
    estimator=MiniBatchKMeans(n_clusters=i)
    estimator.fit(X)
    #Calculate metrics
    calculate_metrics(estimator,X,Y)
    
    # determine predicted labels
    cluster_label = cluster_labels(estimator, Y)
    predicted_Y = infer_data_labels(estimator.labels_, cluster_label)
    
    # calculate and print accuracy
    print('Accuracy: ',metrics.accuracy_score(Y, predicted_Y))
    print()
    

# test kmeans algorithm on testing dataset
# convert each image to 1 dimensional array
X_test = x_test.reshape(len(x_test),-1)
# normalize the data to 0 - 1
X_test = X_test.astype(float) / 255.

#Initialize and Fit KMeans algorithm on training data
kmeans = MiniBatchKMeans(n_clusters = 256)
kmeans.fit(X)
cluster_label = cluster_labels(kmeans, Y)
# predict labels for testing data
test_clusters = kmeans.predict(X_test)
predicted_labels = infer_data_labels(kmeans.predict(X_test), cluster_label)
# calculate and print accuracy
print('Accuracy on test data: ',metrics.accuracy_score(y_test, predicted_labels))
print()


# Initialize and fit KMeans algorithm
kmeans = MiniBatchKMeans(n_clusters = 36)
kmeans.fit(X)
# record centroid values
centroids = kmeans.cluster_centers_
# reshape centroids into images
images = centroids.reshape(36, 28, 28)
images *= 255


# determine cluster labels
cluster_label = cluster_labels(kmeans, Y)

# create figure with subplots using matplotlib.pyplot
fig, axs = plt.subplots(6, 6, figsize = (20, 20))
plt.gray()

# loop through subplots and add centroid images
for i, ax in enumerate(axs.flat):
    
    # determine inferred label using cluster_labels dictionary
    for key, value in cluster_label.items():
        if i in value:
            ax.set_title('Inferred Label: {}'.format(key))
    
    # add image to subplot
    ax.matshow(images[i])
    ax.axis('off')
    
# display the figure
fig.show()
plt.show()
