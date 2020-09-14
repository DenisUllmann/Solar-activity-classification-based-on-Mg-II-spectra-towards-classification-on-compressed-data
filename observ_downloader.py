from irisreader.utils import download
download( "20140910_112825_3860259453", target_directory="." )
from sklearn.datasets import load_iris
data = load_iris()
print(data.target_names)

