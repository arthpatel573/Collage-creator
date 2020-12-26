from flask import Flask, request, jsonify
import os
import subprocess

app = Flask(__name__)

def extract_frames(file_location):
    '''
    Extract video frames and randomly selects 5 frames index
    to create collage

    Args:
        file_location: video file location
    Returns:
        
    '''
    # input and output path
    input = os.path.join(file_location,'minions.mp4')
    output = os.path.join(os.path.join(file_location, 'data'), 'frame_%d.jpg')

    # extract frames if video is available
    if os.path.exists(input):
        query = "ffmpeg -i " + str(input) + " -r 1/1 " + str(output)
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        print(str(response).encode('utf-8'))
    else:
        print('Video seems missing')
        return None

@app.route('/', methods=['POST'])
def collage():
    headers = request.headers
    auth = headers.get("API-Key")
    if auth == 'asoidewfoef':
        request_body = request.get_json(force=True)
        file_location =  request_body['video_location']
        extract_frames(file_location)
        return jsonify({"message": "OK: Frames extracted successfully"}), 200
    else:
        return jsonify({"message": "ERROR: Unauthorized"}), 401

if __name__ == '__main__':
    app.run()
