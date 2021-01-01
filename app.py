from flask import Flask, request, jsonify
import os
import subprocess
import uuid
import random  
from itertools import groupby
from operator import itemgetter
import operator
from sklearn.cluster import KMeans
from sklearn import decomposition
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from skimage import transform
from skimage import io

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

NUM_CLUSTERS = 4

def get_data(folder, video_id):
    '''
    Get video frames data and transform it to required format
    Args:
        folder: frames location folder names
            dtype: str
        video_id: unique id corresponding to the video
            dtype: str
    
    Returns:
        img_data: Array of preprocessed video frames
            dtype: ndarray
    '''
    sorted(os.listdir(folder), key=len)
    img_data = []
    #loops through available image
    for i in range(14):
        filePath = folder + '/'+ video_id +'_' + str(i+1) + '.jpg'
        frame = tf.keras.preprocessing.image.load_img(filePath, target_size=(224, 224))
        # make image data in a compatible form
        processed_frame = np.expand_dims(np.array(frame, dtype= float)/255, axis= 0)
        img_data.append(processed_frame[0])
    img_data = np.asarray(img_data, dtype=np.float32) 
    return img_data

def get_resnet50_features(input, layer_index):
    '''
    Extract features from last layer of ResNet50 model
    Args:
        input: Array of pixels values of the frames
            dtype: ndarray
        layer_index: index of layer for extracting features
            dtype: int
    Returns:
        layer_out: Output from resnet50 model's specified layer
            dtype: ndarray
    '''
    # use pretrained Resnet50 from keras
    model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
    # freeze the layer
    model.trainable = False

    # get features from output layer
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in  model.layers], #call layer output
    )
        
    #extract feature from our dataset
    features = feature_extractor(input)

    #reshape layer output
    layer_out = np.squeeze(features[layer_index]).reshape(input.shape[0], -1)

    return layer_out

def get_closest_points(data, centroid, assigned_points):
    '''
    Args:
        centroid: list of centroid
        assigned_points: list of cluster assignments for each input point
    return:
        closest_list: list of closest point for each cluster
    '''
    closest_list = []
    for cluster in range(NUM_CLUSTERS):
        cluster_points = np.where(assigned_points==cluster)
        if len(cluster_points[0]) == 1:
            closest_list.append(cluster_points[0][0])
        else:
            minimum = 1000
            closest_point = None
            for point in cluster_points[0]:
                euclidean_dist = np.linalg.norm(centroid[cluster]-data[point])
                if minimum > euclidean_dist:
                    minimum = euclidean_dist
                    closest_point = point
            closest_list.append(closest_point)

    closest_list.sort()
    return closest_list

def extract_frames(file_name, video_id):
    '''
    Extract video frames and randomly selects 5 frames index
    to create collage

    Args:
        file_location: video file location
    Returns:
        
    '''
    # input and output path
    input = os.path.join('data', file_name)
    output = os.path.join('data', video_id+'_%d.jpg')

    # extract frames if video is available
    if os.path.exists(input):
        query = "ffmpeg -i " + str(input) + " -r 1/1 " + str(output)
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        print("Response: \n\n", str(response).encode('utf-8'))
    else:
        print('Video seems missing')
        return None

def bound_blur(image, b):
    '''
    Blur the boundaries before putting them into the collage
    '''
    h, w, d = image.shape
    output = np.zeros([h, w, d])
    for i in range(h):
        for j in range(w):
            output[i, j, 0:d] = image[i, j, 0:d] * min(min(i, h - i) / b, 1) * min(min(j, w - j) / b, 1)
    return output


def bound_recover(image, b):
    '''
    Recover the collage's boundary after merged
    '''
    h, w, d = image.shape
    output = np.zeros([h, w, d])
    b_half = int(b / 2)
    for i in range(h):
        for j in range(w):
            output[i, j, 0:d] = image[i, j, 0:d] * max(b / min(i + b_half, h - i + b_half), 1) * max(
                b / min(j + b_half, w - j + b_half), 1)
    return output

def image_merge(images, b, vertical, horizontal):
    '''
    Calculate the pixels and positions of collage
    '''
    n, h, w, d = images.shape
    output = np.zeros([480, 640, d])
    ver = int(vertical * 480)
    hor = int(horizontal * 640)
    assert n == 4, 'image numbers do not match'
    assert b % 2 == 0, 'Bound should be an even number'
    b_half = int(b / 2)

    # transform the original picture into particular size and blur them
    blur_1 = bound_blur(transform.resize(images[0], (ver + b, hor + b, d)), b)
    blur_2 = bound_blur(transform.resize(images[1], (480 - ver + b, hor + b, d)), b)
    blur_3 = bound_blur(transform.resize(images[2], (ver + b, 640 - hor + b, d)), b)
    blur_4 = bound_blur(transform.resize(images[3], (480 - ver + b, 640 - hor + b, d)), b)

    # put them together into one 640*480 image
    output[0:ver + b_half, 0:hor + b_half, 0:d] += blur_1[b_half:ver + b, b_half:hor + b, 0:d]
    output[ver - b_half:480, 0:hor + b_half, 0:d] += blur_2[0:480 - ver + b_half, b_half:hor + b, 0:d]
    output[0:ver + b_half, hor - b_half:640, 0:d] += blur_3[b_half:ver + b, 0:640 - hor + b_half, 0:d]
    output[ver - b_half:480, hor - b_half:640, 0:d] += blur_4[0:480 - ver + b_half, 0:640 - hor + b_half, 0:d]

    return bound_recover(output, b)


@app.route('/', methods=['POST'])
def collage():
    '''
    Create collage from key important frames
    '''
    headers = request.headers
    auth = headers.get("API-Key")

    # Authentication
    if auth == os.getenv('API-Key', ''):
        request_body = request.get_json(force=True)

        # get file name
        file_name =  request_body['file_name']
        video_id = str(uuid.uuid4())
        
        # extract frames from the video
        extract_frames(file_name, video_id)
        img_data = get_data('data', video_id)

        # get output of the last layer
        last_layer_out = get_resnet50_features(img_data, -1)

        # Dimentionality reduction to 3 components
        pca = decomposition.PCA(n_components=3)
        pca.fit(last_layer_out)
        data = pca.transform(last_layer_out)

        # apply kmeans clustering
        kmeans = KMeans(n_clusters=4, random_state=7).fit(data)

        # get indices of important frames of the video
        image_list = get_closest_points(data, kmeans.cluster_centers_, kmeans.labels_)

        # create collage from important frames
        image_full_list = []
        for i in image_list:
            image_full_list.append(os.path.join('data',video_id + '_' +str(i) + '.jpg'))
        imageCollection = io.ImageCollection(image_full_list)
        collage = (image_merge(np.array(list(x for x in imageCollection)), 60, 0.5, 0.5) * 255).astype('uint8')
        # save the collage as merged.jpg
        io.imsave('collage.jpg', collage)

        return jsonify({"message": "OK: Collage saved successfully"}), 200
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401

if __name__ == '__main__':
    app.run()
