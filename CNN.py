import random

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, InputLayer
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory


def dataAugmentationLayer():
    return keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.2)
    ])


def createModelNN1():
    newModel = Sequential()
    newModel.add(Flatten())
    newModel.add(Dense(64, activation='relu'))
    newModel.add(Dense(32, activation='relu'))
    newModel.add(Dense(16, activation='relu'))
    newModel.add(Dense(1, activation='relu'))

    newModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return newModel


def createModelNN2():
    newModel = Sequential()
    newModel.add(Conv2D(16, 3, padding='same', activation='relu'))
    newModel.add(MaxPooling2D())
    newModel.add(Dropout(0.4))
    newModel.add(Conv2D(32, 3, padding='same', activation='relu'))
    newModel.add(MaxPooling2D())
    newModel.add(Dropout(0.4))
    newModel.add(Conv2D(64, 4, padding='same', activation='relu'))
    newModel.add(MaxPooling2D())
    newModel.add(Dropout(0.4))
    newModel.add(Flatten())
    newModel.add(Dense(128, activation='relu'))
    newModel.add(Dropout(0.4))
    newModel.add(Dense(1, activation='sigmoid'))

    newModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return newModel


def createModelNN3():
    newModel = Sequential()
    newModel.add(dataAugmentationLayer())
    newModel.add(Conv2D(16, 3, padding='same', activation='relu'))
    newModel.add(MaxPooling2D())
    newModel.add(Dropout(0.4))
    newModel.add(Conv2D(32, 3, padding='same', activation='relu'))
    newModel.add(MaxPooling2D())
    newModel.add(Dropout(0.4))
    newModel.add(Conv2D(64, 4, padding='same', activation='relu'))
    newModel.add(MaxPooling2D())
    newModel.add(Dropout(0.4))
    newModel.add(Flatten())
    newModel.add(Dense(128, activation='relu'))
    newModel.add(Dropout(0.4))
    newModel.add(Dense(1, activation='sigmoid'))

    newModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return newModel


def fitModel(train_ds, validation_ds, tempEpochs):
    tempModel = createModelNN2()
    # tempModel = load_model('vkrTempFullDense.h5') # vkrTempFullDense vkr vkrTempFirstConv
    newHistory = tempModel.fit(train_ds, validation_data=validation_ds, epochs=tempEpochs)
    # tempModel.save('vkrTempFullDense.h5')

    return tempModel, newHistory


def predictionOutput(prediction):
    if prediction == 0:
        return 'LG'
    else:
        return 'HG'


def predictionInConsole(tempPredictions, validation_ds, count):
    classNames = np.array(validation_ds.class_names)
    for images, labels in validationDataSet1.take(2):
        for i in range(count):
            print(str(i) + ' Label: ' + classNames[labels[i]] + ' value: ' + predictionOutput(tempPredictions[i]))


def predictionsPlot(tempPredictions, count):
    plt.figure(figsize=(10, 10))
    for images, labels in validationDataSet1.take(2):
        for i in range(5,count+5):
            ax = plt.subplot(4, 3, i - 4)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(predictionOutput(tempPredictions[i]))
            plt.axis("off")
    plt.show()


def historyPlot(history):
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([0, 1.1])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')
    plt.show()


def tempHistoryPlot(history1):
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([0, 1.1])
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.title('Точность распознования')
    plt.ylabel('Точность')
    plt.xlabel('Эпохи')
    plt.legend(['Обучение 1', 'Проверка 1', 'Обучение 2', 'Проверка 2', 'Обучение 3', 'Проверка 3'], loc='right')
    plt.show()


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    epochs = 3

    # data
    trainDataSetPath1 = os.path.abspath('C:/datasets/train3')
    # trainDataSetPath2 = os.path.abspath('C:/datasets/train2')
    validationDataSetPath1 = os.path.abspath('C:/datasets/test3')
    # validationDataSetPath2 = os.path.abspath('C:/datasets/test2')
    # validationDataSetPath3 = os.path.abspath('C:/datasets/test3')

    trainDataSet1 = image_dataset_from_directory(
        directory=trainDataSetPath1
    )

    # trainDataSet2 = image_dataset_from_directory(
    #     directory=trainDataSetPath2
    # )

    validationDataSet1 = image_dataset_from_directory(
        directory=validationDataSetPath1,
        shuffle=False
    )

    # validationDataSet2 = image_dataset_from_directory(
    #     directory=validationDataSetPath2,
    #     shuffle=False
    # )
    #
    # validationDataSet3 = image_dataset_from_directory(
    #     directory=validationDataSetPath3,
    #     shuffle=False
    # )

    # model
    model, history1 = fitModel(trainDataSet1, validationDataSet1, epochs)  # , history
    # model, history2 = fitModel(trainDataSet1, validationDataSet2, epochs)  # , history
    # model, history3 = fitModel(trainDataSet2, validationDataSet2, epochs)  # , history
    print(model.summary())
    tempHistoryPlot(history1)

    # # prediction
    # evaluateResult = model.evaluate(validationDataSet3, verbose=0)
    # print('loss: ' + str(evaluateResult[0]) + ' acc: ' + str(evaluateResult[1] * 100) + '%')

    predictions = model.predict(validationDataSet1)
    predictions = predictions.round().astype(np.int64)

    # predictionInConsole(predictions, validationDataSet, 5)  # batch size
    predictionsPlot(predictions, 12)
    plt.show()

    session.close()

