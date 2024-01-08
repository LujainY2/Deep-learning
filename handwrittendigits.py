import os
# computer vision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# dataset

mnist =tf.keras.datasets.mnist
# x image y the number itself (the classification )
# return tow tables with the training & testing
(x_train,y_train),(x_test,y_test) =mnist.load_data()
#Normalize
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model =tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

#optimize the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=15)

model.save('handwritten.model')

# same as just running the same model
##model = tf.keras.models.load_model('handwritten.model')
#loss, accuracy =model.evaluate(x_test,y_test)

#print(loss)
#print(accuracy)

#read pics

import os
for image_number in range(1, 10):
    try:
        img_path = f'digitss/digit{image_number}.png'

        # Check if the image file exists
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)[:, :, 0]
            img = np.invert(np.array([img]))

            # Assuming your model is already defined and loaded
            prediction = model.predict(img)

            print(f"The digit in {img_path} is probably a {np.argmax(prediction)}")

            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        else:
            print(f"Image digit{image_number}.png not found.")

    except Exception as e:
        print(f'Error processing digit{image_number}.png: {e}')

