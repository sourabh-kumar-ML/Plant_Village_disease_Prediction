import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from googlesearch import search 
import webbrowser  
from sklearn.utils import shuffle

EPOCHS = 10  
INIT_LR = 1e-3
BS = 32
image_size = (256, 256)
image_size = 0
root = ''
width=256
height=256
depth=1
#root = "D:\Plant_leave_diseases_dataset_without_augmentation"

disease_dir = listdir(root)   #['Apple___Apple_scab', 'Peach___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Corn___healthy', 'Squash___Powdery_mildew', 'Peach___healthy', 'Tomato___Late_blight', 'Potato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Raspberry___healthy', 'Tomato___Septoria_leaf_spot', 'Tomato___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Grape___Esca_(Black_Measles)', 'Corn___Common_rust', 'Strawberry___healthy', 'Cherry___Powdery_mildew', 'Potato___Early_blight', 'Tomato___Early_blight', 'Cherry___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Corn___Northern_Leaf_Blight', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Grape___Black_rot', 'Pepper,_bell___healthy', 'Strawberry___Leaf_scorch', 'Soybean___healthy', 'Pepper,_bell___Bacterial_spot', 'Background_without_leaves', 'Tomato___Target_Spot', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Tomato_mosaic_virus']

image_list, label_list = [], []

for disease in disease_dir:
    print(f"Processing :: {disease}...")
    leaf_images = listdir(f"{root}/{disease}")
    counter = 0
    for img in leaf_images:
        image = cv2.imread(f"{root}/{disease}/{img}")
        
        if image is not None :
          image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          image = cv2.resize(image, (256,256))   
          image_list.append(img_to_array(image))
          label_list.append(disease)
          counter += 1
        if counter == 500:
            break
print("Done Processing")

len(image_list)

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
classes = label_binarizer.classes_

#feature_scaling
std_image_list =np.array(image_list,dtype =np.float16)/ 255.0

#Splitting_data 
X_train, X_test, Y_train, Y_test = train_test_split(std_image_list, image_labels, test_size=0.2, random_state = 0,stratify = label_list) 
X_train, Y_train = shuffle(X_train, Y_train)
X_test,Y_test = shuffle(X_test,Y_test)

print(label_binarizer.classes_)

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))

model.summary()

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")

history = model.fit_generator(
    aug.flow(X_train, Y_train, batch_size=BS),
    validation_data=(X_test, Y_test),
    steps_per_epoch=len(X_train) // BS,
    epochs=EPOCHS, verbose=1
    )
# This block trains the model

# save the model to disk
print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

print("[INFO] Calculating model accuracy")
scores = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {scores[1]*100}")

def pred_disease():
    adr = input("Enter adress of plant leave:")
    image = cv2.imread(f'{adr}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256,256))
    std_img = np.array(image, dtype = np.float16)/255.0
    std_img = std_img.reshape(-1,256,256,1)
    out = model.predict(std_img).ravel()
    diseas = classes[out.argmax()]
    diseas = diseas.replace('_'," ")
    print('plant is suffuering from :: {}'.format(diseas))
    diseas = diseas.replace('_'," ")
    query = diseas +"Treatment and cure"
    data = search(query, tld="co.in", num=10, stop=1, pause=2)
    url = ''
    for j in search(query, tld="co.in", num=10, stop=1, pause=2): 
        url = j
    webbrowser.open(url, new=0, autoraise=True)

#to be used only when training is completed i.e load trained model
with open('cnn_model.pkl','rb') as f:
    model = pickle.load(f)
with open('label_transform.pkl','rb') as d:
    labael_binarizer = pickle.load(d)
classes = labael_binarizer.classes_

pred = True
while pred:
    x = int(input('\n\n1.Enter 1 to Predict Disease\n2.Enter 2 to Exit\n'))
    if x== 1:
        pred_disease()
    elif x == 2:
        pred = False
