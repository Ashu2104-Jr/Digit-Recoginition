from PIL import Image
import numpy as np
import os
gradientList=[]
progesslist="----------------------------------------------------------------------------------------------------"
percent=0
def print1():
    print(f"[{progesslist}]{percent}%")
def changelist():
    global progesslist
    progesslist="#"*percent+progesslist[percent:]
def relu(x):
    return np.maximum(0,x)
def relu_dervi(x):
    return (x>0).astype(float)
def softmax(x):
    x=x-np.max(x,axis=1,keepdims=True)
    ex=np.exp(x)
    return ex/np.sum(ex,axis=1,keepdims=True)
def getSetArray(val):
    array=[0,0,0,0,0,0,0,0,0,0]
    array[val]=1
    return array

def pixeltoimg(input):
    inputList=[]
    for i in range(10):
        folder_path=f"/home/jarvis/DS_akrsir/images/{input}/{i}"
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.png'):
                    img_path = os.path.join(folder_path, file_name)
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img).flatten()
                    img_array=[x / 255 for x in img_array]
                    inputList.append(img_array)
                    gradientList.append(getSetArray(i))
        global percent
        percent=percent+4
        os.system('clear')
        changelist()
        print1()
    return inputList
def train(inputList):
    global gradientList
    w1=np.random.randn(784,392)*0.01
    w2=np.random.randn(392,196)*0.01
    w3=np.random.randn(196,98)*0.01
    w4=np.random.randn(98,10)*0.01

    b1= np.zeros((392,))
    b2= np.zeros((196,))
    b3= np.zeros((98,))
    b4= np.zeros((10,))
    gradientList=np.array(gradientList)
    inputList=np.array(inputList)
    for i in range(15):
        shuffleListIndex=np.random.permutation(len(inputList))
        inputList=inputList[shuffleListIndex]
        gradientList=gradientList[shuffleListIndex]
        for j in range(0,len(inputList),40):
            batch=inputList[j:j+40]
            gradientBatch=gradientList[j:j+40]
            layer1=np.dot(batch,w1)+b1
            layer1=relu(layer1)
            layer2=np.dot(layer1,w2)+b2
            layer2=relu(layer2)
            layer3=np.dot(layer2,w3)+b3
            layer3=relu(layer3)
            output=np.dot(layer3,w4)+b4
            
            #backpropogation
            gDescent=(softmax(output)-gradientBatch)/40

            dw4=np.dot(layer3.T,gDescent)
            db4=np.sum(gDescent,axis=0)
            gDescent=np.dot(gDescent,w4.T)*relu_dervi(layer3)
            
            dw3=np.dot(layer2.T,gDescent)
            db3=np.sum(gDescent,axis=0)
            gDescent=np.dot(gDescent,w3.T)*relu_dervi(layer2)

            dw2=np.dot(layer1.T,gDescent)
            db2=np.sum(gDescent,axis=0)
            gDescent=np.dot(gDescent,w2.T)*relu_dervi(layer1)

            dw1=np.dot(batch.T,gDescent)
            db1=np.sum(gDescent,axis=0)
            
            w4-=0.05*dw4
            b4-=0.05*db4
            w3-=0.05*dw3
            b3-=0.05*db3
            w2-=0.05*dw2
            b2-=0.05*db2
            w1-=0.05*dw1
            b1-=0.05*db1
        global percent
        percent=percent+4
        os.system('clear')
        changelist()
        print1()
    return w1,w2,w3,w4,b1,b2,b3,b4

os.system('clear')
print1()
inputList=pixeltoimg("training")
w1,w2,w3,w4,b1,b2,b3,b4=train(inputList)

# Save weights to file
np.savez('weights.npz', w1=w1, w2=w2, w3=w3, w4=w4, b1=b1, b2=b2, b3=b3, b4=b4)
print("Weights saved to weights.npz")

