# Collage-creator
[![Python Version](https://img.shields.io/badge/python-3.7+-blue?logo=python)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Flask API to create collage from the video. The collage is formed by extracting important frames of the video.

The important frames selection task is performed based on the features of ResNet model. The dimensionalities of these features are reduced using Principal Component Analysis (PCA). Finally, after performing K-means clustering on these features, 4 important frames are chosen from the video.


## Installation:

Move the video inside `/data` folder from the root folder. 

Install the packages

```
pip install -r requirements.txt
```

## Running the API:

Create `.env` file as per the `.env-example`

Run the Flask endpoint:

```
python main.py
```

This API will run at port 5000. 

Alternatively, you can build the docker container image and run the docker container.

```
docker build -t collage-creator .
docker run -dp 5000:5000 collage-creator
```

### Request format:
```
{
    "file_name": "FILE_NAME.mp4"
}
```

The collage will be downloaded in root folder.

### Code Style:

Black, the uncompromising code formatter, is used to format the code, so there is no need to worry about style. Beyond that, where reasonable, for example for docstrings, the **Google Python Style Guide** is followed.

