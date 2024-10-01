# Import libraries
import os
from urllib.request import urlretrieve
import argparse


# Define the function to download the model
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--model', 
    choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11b', 'yolo11l', 'yolo11x'],
    default='yolo11n', 
    help='Model to download'
)

args = parser.parse_args()

def download_model(model):
    '''
        Function to download the model from the github release page
    '''

    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/" + model + ".pt"
    # Downloading zip file using urllib package.
    print("Downloading the model...")
    urlretrieve(url, model + ".pt")
    print("Model downloaded successfully!")


# Call the function to download the model
download_model(args.model)