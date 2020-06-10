import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)

mnist = tf.keras.datasets.mnist     # 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

# Adding layers to model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Adding some parameters to give information about type of training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# To test the loss and accuracy over test data
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# To save and reload the model
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')


predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))


plt.imshow(x_test[0])
plt.show()




