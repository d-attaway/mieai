""" Data handling functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import requests
from zipfile import ZipFile
from io import BytesIO
import os
import shutil

def get_models(data_location):
    '''
    Download and unzip AI old_mieai_models from Zenodo
    '''
    # Zenodo link
    url = 'https://zenodo.org/records/20346256/files/models.zip?download=1'

    # download and unzip folder from Zenodo
    os.makedirs(data_location, exist_ok=True)
    r = requests.get(url)
    ZipFile(BytesIO(r.content)).extractall(data_location)

    # move files out of models folder
    models_folder = os.path.join(data_location, 'models')
    for f in os.listdir(models_folder):
        shutil.move(os.path.join(models_folder, f), data_location)
    shutil.rmtree(models_folder)

    # delete MACOSX folder
    shutil.rmtree(os.path.join(data_location, '__MACOSX'))

    return data_location