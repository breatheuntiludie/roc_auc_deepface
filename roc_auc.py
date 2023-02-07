import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_pickle('C:/Users/user/Desktop/roc/SFace.pkl')
print(pd.DataFrame.max(df))
k = 50
threshold = 0
step = 0.001
roc = []
roc.append([])
roc.append([])
roc.append([])
while threshold <= 1:
    counter_tpr = 0
    counter_fpr = 0
    roc[0].append(round(threshold,2))
    for i in range(k):
        for j in range(k):
            if i == j and  df[i][j]/17.270312 <= threshold:
                counter_tpr += 1
            elif i != j and df[i][j]/17.270312 <= threshold:
                counter_fpr += 1
    roc[1].append(counter_tpr/k)
    roc[2].append(counter_fpr/(k*k - k))
    threshold = threshold + step

counter = 0
auc = 0
while counter < (1/step)-1:
    square = roc[1][counter]*(roc[2][counter+1]-roc[2][counter])
    triangle = (roc[1][counter+1] - roc[1][counter])*(roc[2][counter+1]-roc[2][counter])
    auc = auc + square + triangle
    counter = counter + 1

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-curve(SFace), AUC = ' + str(auc))
plt.plot(roc[2],roc[1])
plt.grid(True)
plt.savefig('C:/Users/user/Desktop/roc/SFace.png')
plt.show()