import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

#Digunakan untuk membaca data
def readData(nama):
    data = pd.ExcelFile(nama)
    latih = pd.read_excel(data, 'train')
    tes = pd.read_excel(data, 'test')
    return latih, tes

#Digunakan untuk menulis output file
def writeData(nama, output):
    inp = pd.ExcelFile(nama)
    data = pd.read_excel(inp, 'test')
    id = data['id'].tolist()
    x1 = data['x1'].tolist()
    x2 = data['x2'].tolist()
    x3 = data['x3'].tolist()

    df = pd.DataFrame({'id': id,'x1': x1, 'x2': x2, 'x3': x3, 'y': output })
    df.to_excel('output.xlsx', sheet_name='hasil', index=False)
#Karena KNN Tidak ada Training Model maka langsung melakukan uji

#Untuk menghitung Jarak
def dist(Train, Test):
    distdmp = []
    for j in range(len(Test[0])):
        dmp = []
        for i in range(len(Train[0])):
            a = (Train[0][i], Train[1][i], Train[2][i])
            b = (Test[0][j], Test[1][j], Test[2][j])
            dmp.append(euclidean(a,b))
        distdmp.append(dmp)
    return distdmp

#Untuk menentukan Class
def pick_Class(data, Train, k):
    dataTrain = Train
    Class = []
    for i in range(len(data)):
        x = data[i]
        dmpList = [x for _, x in sorted(zip(x,dataTrain))]
        kelas = max(set(dmpList[0:k]), key = dmpList[0:k].count)
        Class.append(kelas)
    return Class

#Evaluasi Model
def accurate(k):
    data = pd.ExcelFile('traintest.xlsx')
    latih = pd.read_excel(data, 'train')
    x1 = latih['x1'].tolist()
    x2 = latih['x2'].tolist()
    x3 = latih['x3'].tolist()
    y = latih['y'].tolist()
    print("Uji coba Evaluasi dengan tetangga K:", k)
    print("Evaluasi Pertama")
    x1Eva1 = x1[0:74]
    x2Eva1 = x2[0:74]
    x3Eva1 = x3[0:74]
    x1Eva1U1 = x1[74:148]
    x1Eva1U2 = x2[74:148]
    x1Eva1U3 = x3[74:148]
    yEva1 = y[74:148]
    arrayEva1 = np.array((x1Eva1, x2Eva1, x3Eva1), dtype = int)
    arrayEva2 = np.array((x1Eva1U1, x1Eva1U2, x1Eva1U3), dtype = int)
    dmp1 = dist(arrayEva1, arrayEva2)
    Kelas1 = pick_Class(dmp1, yEva1, k)
    count = 0
    for i in range(len(dmp1)):
        if Kelas1[i] == yEva1[i]:
            count += 1
    print("Tingkat Error Evaluasi: ", 1 - (count/len(yEva1)))

    print("Evaluasi Kedua")
    x1Eva1 = x1[148:222]
    x2Eva1 = x2[148:222]
    x3Eva1 = x3[148:222]
    x1Eva1U1 = x1[222:296]
    x1Eva1U2 = x2[222:296]
    x1Eva1U3 = x3[222:296]
    yEva1 = y[222:296]
    arrayEva1 = np.array((x1Eva1, x2Eva1, x3Eva1), dtype = int)
    arrayEva2 = np.array((x1Eva1U1, x1Eva1U2, x1Eva1U3), dtype = int)
    dmp1 = dist(arrayEva1, arrayEva2)
    Kelas1 = pick_Class(dmp1, yEva1, k)
    count = 0
    for i in range(len(dmp1)):
        if Kelas1[i] == yEva1[i]:
            count += 1
    print("Tingkat Error Evaluasi: ", 1 - (count/len(yEva1)))

            

def main():
    xTrain, yTest = readData('traintest.xlsx')
    arrayTrain = np.array((xTrain['x1'].tolist(),xTrain['x2'].tolist(),xTrain['x3'].tolist()), dtype= int)
    arrayTest = np.array((yTest['x1'].tolist(),yTest['x2'].tolist(),yTest['x3'].tolist()), dtype= int)
    dmp = dist(arrayTrain, arrayTest)
    k = int(input("Masukkan banyak k(Tetannga) yang diinginkan: "))
    dataTrain = xTrain['y'].tolist()
    Kelas = pick_Class(dmp, dataTrain, k)
    accurate(k)
    writeData('traintest.xlsx', Kelas)

if __name__ == '__main__':
    print("Tugas=03-Learning")
    print("===================")
    main()
    print("\nProgram telah selesai mengeluankan output dengan nama file output.xlsx")
    print("===================")
    print("Program Selesai")


