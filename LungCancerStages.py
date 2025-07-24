import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from tkinter import filedialog, Text
import numpy as np
import cv2
import pickle
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from keras.models import Sequential
from keras.models import Sequential, load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Disable oneDNN optimizations


# Create the main application window
main = tk.Tk()
main.title("Lung Cancer Stages Prediction")
main.geometry("1000x650")

global filename
global classifier
global X_train, y_train, X_test, y_test

labels = ['Normal', 'Stage1', 'Stage2', 'Stage3']

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', tk.END)
    text.insert(tk.END, filename + ' Loaded\n\n')
    text.insert(tk.END, "Stages Found in Dataset: " + str(labels) + "\n\n")
    print("Dataset uploaded:", filename)

def getID(name):
    index = labels.index(name) if name in labels else -1
    print("Getting ID for:", name)
    return index

def preprocessDataset():
    text.delete('1.0', tk.END)
    global filename, X, Y
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        print("Loaded existing dataset.")
    else:
        X, Y = [], []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(os.path.join(root, directory[j]))
                    img = cv2.resize(img, (32, 32))
                    im2arr = np.array(img).reshape(32, 32, 3)
                    X.append(im2arr)
                    label = getID(name)
                    Y.append(label)
                    print(name, "label:", label)

        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt', X)
        np.save('model/Y.txt', Y)

    X = X.astype('float32') / 255
    text.insert(tk.END, "Dataset Preprocessing Completed\n")
    text.insert(tk.END, "Total images found in dataset: " + str(X.shape[0]) + "\n\n")
    text.update_idletasks()
    img = cv2.resize(X[0], (100, 100))
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)

def trainTest():
    text.delete('1.0', tk.END)
    global filename, X, Y, X_train, y_train, X_test, y_test
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(tk.END, "80% images used to train CNN Algorithm: " + str(X_train.shape[0]) + "\n")
    text.insert(tk.END, "20% images used to test CNN Algorithm: " + str(X_test.shape[0]) + "\n")

def runCNN():
    text.delete('1.0', tk.END)
    global X_train, y_train, X_test, y_test, classifier
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(units=y_train.shape[1], activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_path = "model/model_weights.hdf5"
    if not os.path.exists(model_path):
        model_check_point = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
        hist = classifier.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        with open('model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    else:
        classifier = load_model(model_path)

    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100

    text.insert(tk.END, "CNN Lung Cancer Accuracy: " + str(a) + "\n")
    text.insert(tk.END, "CNN Lung Cancer Precision: " + str(p) + "\n")
    text.insert(tk.END, "CNN Lung Cancer Recall: " + str(r) + "\n")
    text.insert(tk.END, "CNN Lung Cancer F1 Score: " + str(f) + "\n\n")

    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title("CNN Confusion Matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def graph():
    with open('model/history.pckl', 'rb') as f:
        graph = pickle.load(f)
    accuracy = graph['val_accuracy']
    loss = graph['val_loss']

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color='green')
    plt.plot(loss, 'ro-', color='red')
    plt.legend(['CNN Accuracy', 'CNN Loss'], loc='upper left')
    plt.title('CNN Lung Cancer Training Accuracy & Loss Graph')
    plt.show()

def predict():
    global classifier, labels
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img).reshape(1, 32, 32, 3)
    img = im2arr.astype('float32') / 255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.resize(image, (700, 400))
    cv2.putText(img, 'Lung Cancer Predicted as: ' + labels[predict], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Lung Cancer Predicted as: ' + labels[predict], img)
    cv2.waitKey(0)

def close():
    main.destroy()

# Create the UI elements
font = ('times', 15, 'bold')
title = tk.Label(main, text='Lung Cancer Stages Prediction', justify=tk.LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=100, y=5)

font1 = ('times', 12, 'bold')
uploadButton = tk.Button(main, text="Upload Lung Cancer Stages Dataset", command=uploadDataset)
uploadButton.place(x=10, y=100)
uploadButton.config(font=font1)

preprocessButton = tk.Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=300, y=100)
preprocessButton.config(font=font1)

splitButton = tk.Button(main, text="Split Dataset Train Test", command=trainTest)
splitButton.place(x=480, y=100)
splitButton.config(font=font1)

cnnButton = tk.Button(main, text="Run CNN Algorithm", command=runCNN)
cnnButton.place(x=710, y=100)
cnnButton.config(font=font1)

graphButton = tk.Button(main, text="CNN Training Graph", command=graph)
graphButton.place(x=10, y=150)
graphButton.config(font=font1)

predictButton = tk.Button(main, text="Predict Lung Cancer Stage", command=predict)
predictButton.place(x=300, y=150)
