from flask import Flask, request, jsonify
import os
import subprocess
import uuid

app = Flask(__name__)

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
        return jsonify({"message": "OK: Frames extracted successfully"}), 200
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401

if __name__ == '__main__':
    app.run()
