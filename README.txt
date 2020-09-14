This several script combined with train dataset will help you to reproduce results of arXiv:

Scripts have following purposes:

observation2centroids.py - downloads and saves selected observation with irisreader into local folder, tranforms data into clusterized centroids format (compressed data)
observ_downloader.py - downloads and saves selected observation with irisreader into local folder

xgboost_classfication.py - trains xgboost ML model on the train compressed dataset
cnn_classfication.py - trains CNN ML model on the train compressed dataset
xgboost_rawclassfication.py - xgboost ML model on the train raw dataset
tools.py - just some additional functions

One should have irisreader, xgboost, sklearn and pandas packages install in order to work properly for this scripts.

To get all the required packages, you can also run pip install -r requirements.txt