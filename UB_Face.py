'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))

    faces_rec = face_cascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors=6,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in faces_rec:
        cv2.rectangle(img, (x,y,w,h), (255, 0, 0), 2)
        detection_results.append([float(x), float(y), float(w), float(h)])
    return detection_results

def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
     # Preprocessing Images. 
    a = split_label_images(imgs)
    images = a[0]
    labels = a[1]
    crop_img = []
    enc_img = []
   
    # Find bounding box.
    bbox = []
    frame = []
    real_img=[]
    for names in range(len(images)):
        read_image = images[names]
        real_img.append(read_image)

        face1 = cascad(read_image)        

        top = face1[0][1]
        left = face1[0][0]
        bottom = face1[0][0] + face1[0][2]
        right = face1[0][1] + face1[0][3]
        boxes = [(top,bottom,right,left)]
        cut = real_img[names][top:right, left:bottom]

        frame.append(cut)
        
        face_enc=face_recognition.face_encodings(read_image, boxes)
        enc_img.append(np.array(face_enc))

        crop_img.append(cv2.resize(cut, (128, 128)))

        bbox.append(face_enc[0])
    #print(bbox2)
    #print(frame)
    # Clustering begins
    bbox = np.array(bbox)
    clusters_res = cluster(bbox, int(K))

    labels = list(imgs.keys())
    
    unique_clusters = np.unique(clusters_res)
    ina = []
    result = {}
    counter = 0
    for i in unique_clusters:
        result[i] = list()
    for label in unique_clusters:
        temp_list=[]
        index = np.where(clusters_res == label)[0]
        ina.append(index)
   
        for j in index:
            temp_list.append(labels[j])
        cluster_results[counter] = temp_list
        counter+=1
    #print(clusters_res)
    #print(cluster_results)
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)
def cascad(img):
    detection_results = []

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, 
        'haarcascade_frontalface_default.xml'))

    faces_rec = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces_rec:
        #cv2.rectangle(img, (x,y,w,h), (255, 0, 0), 2)
        detection_results.append([int(x), int(y), int(w), int(h)])
    return detection_results

def split_label_images(imgs):
    images = []
    labels = {}
    counter = 0
    for key in imgs:
        labels.update({counter: key})
        images.append(imgs[key])
        counter+=1
    a = (images, labels)
    return a

def euclidian_distance(point1, point2): # find distance.
    return np.sqrt(np.sum((point1 - point2)**2))

def initialize_centroid(image, K):
    num_imgs = image.shape[0]

    list_centroid = []

    random_val = np.random.randint(num_imgs)
    list_centroid.append(image[random_val, :])

    for _d in range(K-1):
        distance_list = []
        for img_idx in range(num_imgs):
            img_point = image[img_idx, :]
            max_distance = np.Inf

            for each_centroid in list_centroid:
                new_dist = euclidian_distance(img_point, each_centroid)

                max_distance = min(max_distance, new_dist)
            distance_list.append(max_distance)
        distance_list = np.array(distance_list)
        max_point = np.argmax(distance_list)
        new_centroid = image[max_point, :]
        list_centroid.append(new_centroid)
    return list_centroid

def final_clust_calc(clusters, image):
    n, _n = image.shape
    final_val = np.empty(n)

    for clust_index, cluster_val in enumerate(clusters):
        for i in cluster_val:
            final_val[i] = clust_index
    return final_val

def cluster(image, K):
    num_epochs = 10
    centroid_list = initialize_centroid(image, K)

    for _iteration in range(num_epochs):

        cluster_list = []
        # empty list of lists.
        for i in range(K):
            l = []
            cluster_list.append(l)

        for index, value in enumerate(image):
            centroid_indices = np.argmin([euclidian_distance(value, point) for point in centroid_list])
            cluster_list[centroid_indices].append(index)

        centroid_list_temp = centroid_list
        len = image.shape[1]
        current_centroid = np.zeros((K, len))

        for index, cluster in enumerate(cluster_list):
            current_centroid[index] = np.mean(image[cluster], axis=0)
        distance_list = []

        for i in range(K):
            temp = euclidian_distance(centroid_list_temp[i], current_centroid[i])
            distance_list.append(temp)
        if sum(distance_list) == 0:
            break
    return final_clust_calc(cluster_list, image)

"""
Reference: 
1. https://www.researchgate.net/publication/277929875_System_Architecture_for_Real-Time_Face_Detection_on_Analog_Video_Camera
2. https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
3. https://face-recognition.readthedocs.io/en/latest/index.html
4. https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
5. https://medium.com/data-folks-indonesia/step-by-step-to-understanding-k-means-clustering-and-implementation-with-sklearn-b55803f519d6

"""