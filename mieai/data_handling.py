""" Data handling functionalities """
# pylint: disable=C0415,R0902,R0912,R0914,R0915

import requests
from zipfile import ZipFile
from io import BytesIO

def get_models(url):
    '''
    Download and unzip AI models from Zenodo
    '''
    # url = 'https://zenodo.org/records/20346256/files/models.zip?download=1'
    folder_name = 'models'
    r = requests.get(url)
    ZipFile(BytesIO(r.content)).extractall(folder_name)