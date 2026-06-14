import urllib.request
import os

os.makedirs("data", exist_ok=True)

url = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt"
print("Downloading NSL-KDD dataset...")
urllib.request.urlretrieve(url, "data/KDDTrain+.txt")
print("Done! File saved to data/KDDTrain+.txt")

url2 = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest+.txt"
print("Downloading test set...")
urllib.request.urlretrieve(url2, "data/KDDTest+.txt")
print("Done! Both files downloaded.")