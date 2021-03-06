import tensorflow as tf
from progress.bar import Bar
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

split = 0.90
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

trainx = features[:splitNum]
trainy = labels[:splitNum]
testx = features[splitNum:]
testy = labels[splitNum:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(featureCount,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(trainx, trainy, epochs = 10)
model.evaluate(testx,  testy, verbose = 2)
