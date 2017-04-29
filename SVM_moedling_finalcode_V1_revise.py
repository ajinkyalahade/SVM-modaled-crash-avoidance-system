import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt  #importing the required tools and algorithms

directory1 = 'A:\PERO'
directory2 = 'A:\PERO\PERO 0'  #add path for the files PER files obtained from MATLAB

directories = []
mean_arr = []
YT_df2 = []
camp_arr = []
PER = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  #Setting a range for the PER values

os.chdir(directory2)
for files in glob.glob("*.csv"):
    data1 = pd.read_csv(files, delimiter=',', header=None)
    cars = pd.DataFrame(data1)
    YT = cars.iloc[:, 4]
    YT_df = pd.DataFrame(YT)
    YT_df2.append(YT_df)

def AJ_algorithm(directory):      #seeting directory and defining variables
    print directory
    acc2 = []
    df2 = []
    acc4 = []
    for files2 in glob.glob("*.csv"):
        data = pd.read_csv(files2, delimiter=',', header=None)
        df = pd.DataFrame(data)
        df2.append(df)

    for p in range(0, 823, 1):    #declaring the files to be scanned
        df3 = pd.concat([df2[p], YT_df2[p]], axis=1, ignore_index=True)

        x1 = df3.iloc[:, 0:4]
        y = df3.iloc[:, 4]
        z = df3.iloc[:, 5]

        x1_train, x1_test, y_train, y_test, z_train, z_test = train_test_split(x1, y, z, test_size=0.3, random_state=5)

        acc3 = accuracy_score(z_test, y_test)
        acc4.append(acc3)

        a = len(np.unique(y_train))
        if a == 1:
            continue
        else:

            clf = SVC()
            clf.fit(x1_train, y_train)
            pred = clf.predict(x1_test)
            acc = accuracy_score(z_test, pred)
            acc2.append(acc)
    mean_arr.append(np.mean(acc2))
    camp_arr.append(np.mean(acc4))

directories = [os.path.abspath(x[0]) for x in os.walk(directory1)]
directories.remove(os.path.abspath(directory1))

for i in directories:
    os.chdir(i)
    AJ_algorithm(i)

print "Mean accuracy of CAMPLinear: ", camp_arr
print "Mean accuracy of my algorithm: ", mean_arr


#poltting the graph results
plt.figure()
plt.plot(PER, mean_arr, label='SVM', marker='8', color='gold')
plt.plot(PER, camp_arr, label='CAMPLinear', marker='8', color='black')
plt.title('Accuracy Vs. PER Plot')
plt.xlabel('Packet Error Rate')
plt.ylabel('Accuracy')
plt.legend(loc=1)
plt.show()
