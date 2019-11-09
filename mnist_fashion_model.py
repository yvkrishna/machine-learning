import tensorflow as tf
# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset,metadata=tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset,test_dataset=dataset['train'],dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

num_training_sets=metadata.splits['train'].num_examples;
num_testing_sets=metadata.splits['test'].num_examples;
print(num_training_sets,num_testing_sets)

def normalize(images,labels):
  images=tf.cast(images,tf.float32)
  images/=255
  return images,labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset=train_dataset.map(normalize);
test_dataset=test_dataset.map(normalize);

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset=train_dataset.cache()
test_dataset=test_dataset.cache()

for image,label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()

l0=tf.keras.layers.Flatten(input_shape=(28, 28, 1))
l1=tf.keras.layers.Dense(128,activation=tf.nn.relu)
l2=tf.keras.layers.Dense(10, activation=tf.nn.softmax)

model=tf.keras.Sequential([l0,l1,l2]);
print(type(model))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset=train_dataset.repeat().shuffle(num_training_sets).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_training_sets/BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_testing_sets/32))
print('Accuracy on test dataset:', test_accuracy)

for test_images,test_labels in test_dataset.take(1):
  test_images=test_images.numpy()
  test_labels=test_labels.numpy()
  predictions = model.predict(test_images)