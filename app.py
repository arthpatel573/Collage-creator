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


app = Flask(__name__)

NUM_CLUSTERS = 4

def get_data(folder, video_id):
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

def resnet50(input, layer_index):
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

@app.route('/', methods=['POST'])
def collage():
    headers = request.headers
    auth = headers.get("API-Key")
    if auth == 'asoidewfoef':
        request_body = request.get_json(force=True)
        file_name =  request_body['file_name']
        video_id = str(uuid.uuid4())
        extract_frames(file_name, video_id)
        img_data = get_data('data', video_id)
        # get output of the last layer
        last_layer_out = resnet50(img_data, -1)

        pca = decomposition.PCA(n_components=3)
        pca.fit(last_layer_out)

        data = pca.transform(last_layer_out)

        kmeans = KMeans(n_clusters=4, random_state=7).fit(data)

        closest_points = get_closest_points(data, kmeans.cluster_centers_, kmeans.labels_)

        return jsonify({"message": "OK: Important frames are "+ str(closest_points)}), 200
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401

if __name__ == '__main__':
    app.run()
