import tensorflow as tf
from progress.bar import Bar
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# file = "D:/lurd-pca/restored-16000-8000-16000-99.86.txt"
# featureCount = 16000
# samples = 9452

file = "D:/lurd-pca/reduced-16000-2000-74.36.txt"
featureCount = 2000
samples = 9452

# file = "trainingData.txt"
# featureCount = 2000
# samples = 10983

split = 0.95
splitNum = int(samples * split)

print("Opening and reading data file...")
with open(file) as lurdFile:
    text = lurdFile.read()
    text = text.split('\n')
    labels = []
    features = []
    with Bar('Reading', max=samples) as bar:
        for LableIndex in range(samples):
            startIndex = LableIndex * featureCount + LableIndex
            if(text[startIndex] == 'left'):
                labels.append(0)
            if(text[startIndex] == 'right'):
                labels.append(1)
            if(text[startIndex] == 'up'):
                labels.append(2)
            if(text[startIndex] == 'down'):
                labels.append(3)
            features.append([])
            for valueIndex in range(featureCount):
                features[LableIndex].append(float(text[startIndex+valueIndex+1]))
                pass
            bar.next()
            pass

trainx = np.expand_dims(np.array(features[:splitNum]), axis = 2)
trainy = np.expand_dims(np.array(labels[:splitNum]), axis = 1)
testx  = np.expand_dims(np.array(features[splitNum:]), axis = 2)
testy  = np.expand_dims(np.array(labels[splitNum:]), axis = 1)

print(trainx.shape)
print(trainy.shape)
print(testx.shape)
print(testy.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=( 2000, 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainx, trainy, epochs = 10)
model.evaluate(testx,  testy, verbose = 2)
