import tensorflow as tf
from progress.bar import Bar


file = "D:/lurd-pca/reduced-16000-2000-74.36.txt"
featureCount = 2000
samples = 9452

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

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2000,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(features, labels, epochs=10)

# model.evaluate(x_test,  y_test, verbose=2)
