import pandas as pd
from deepface import DeepFace
import time

# we will use this models in our task
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace",]
# we have 3 different metrics, but we use only cosine
metrics = ["cosine", "euclidean", "euclidean_l2"]

pathes = [] # list of pathes to photo
for i in range(50):
    pathes.append([])
    path1 = "C:/Users/user/Desktop/roc/bd/" + str(i) + "/1.jpg" #chande adress for current db
    path2 = "C:/Users/user/Desktop/roc/bd/" + str(i) + "/2.jpg" #chande adress for current db
    pathes[i].append(path1)
    pathes[i].append(path2)

compare = [] # massive of our compare
k = 50
for i in range(k):
    compare.append([])
    for j in range(k):
        compare[i].append(j)

for i in range(k):
    print(i,'\n')
    for j in range(k):
        timer_begin = time.time()
        result = DeepFace.verify(img1_path = pathes[j][1], img2_path = pathes[i][0], model_name = models[1], distance_metric = metrics[1])
        timer_end = time.time()
        timer_secs = timer_end - timer_begin
        times_msecs = timer_secs * 1000
        print("Затраченное время: %f" ,times_msecs)
        compare[i][j] = result['distance']

df = pd.DataFrame(compare)
df.to_pickle('C:/Users/user/Desktop/roc/name.pkl') #change with CNN from model