# Face recognition Final Project : 

Credit
https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

CR Template
https://laurentnajman.org/uploads/images/DeepLearning/IEEE_Report_Template.pdf


# Architechture : 
**Remember that you must ALWAYS normalize your data.**

#### DataSet : 
$70$ % train set -- $30%$ % test set


### details : 

**Prevent over-fitting**  regularization techniques ? 
1. Early stopping 
    `model.fit(x_train, y_train, epochs=4, batch_size=512)`
3. Norm penalization
    `model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))`
3. [dropout](https://keras.io/api/layers/regularization_layers/dropout/)
    `model.add(layers.Dropout(0.5))`
4. freezing

**Type of *loss***
1. loss='binary_crossentropy'
2. loss='categorical_crossentropy'

**Type of *activation***
1. `'relu'`
2. `'sigmoide'`
3. `'soft-max'`


**Save the model ?**
```python
import tensorflow as tf
import gc

tf.keras.backend.clear_session()
del model
gc.collect()
```
**Summary of a model**
`model.summary()`

#### Convnet Note : 
A ConvNet takes as input a tensor of shape `(image_height, image_width, image_channels)`
```python
model = models.Sequential()
model.add( layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim) )
model.add( layers.MaxPooling2D((2, 2)) )
model.add( layers.Conv2D(64, (3, 3), activation='relu') )
model.add( layers.MaxPooling2D((2, 2)) )
model.add( layers.Conv2D(64, (3, 3), activation='relu') )
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
```
> The next step is to feed our last layer's output, a tensor of shape (3, 3, 64), into a fully-connected classifier. However, such a classifier processes 1D vectors, whereas our current output is a 3D tensor. So first, we have to flatten our 3D outputs to 1D, and then add a few Dense layers on top. We are going to do 10-way classification, so we use a final layer with 10 outputs and a softmax activation.

##### Process images : 
```Python
from keras.preprocessing import image

# All images will be rescaled by 1./255
train_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,           # This is the directory where the images are stored
        target_size=(128, 128), # All images will be resized to 128x128
        batch_size=20,
        class_mode='binary'     # Since there are only two classes, we need binary labels
)
```

> Like in the previous section, our convnet is a stack of alternated Conv2D (with relu activation) and MaxPooling2D layers. However, since we are dealing with bigger images and a more complex problem, we will make our network accordingly larger: it will have one more Conv2D + MaxPooling2D stage. This serves both to augment the capacity of the network, and to further reduce the size of the feature maps, so that they aren't overly large when we reach the Flatten layer. Since we are attacking a binary classification problem, we end the network with a single unit and a sigmoid activation. This unit will encode the probability that the network is looking at one class or the other.
```python
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-4), metrics=['acc'])
```

**Save the model** 
```
model.save('models/cats_and_dogs_128x128_1.h5')
model = models.load_model('models/cats_and_dogs_128x128_1.h5')
```
#### freesing detail 
>Another widely used technique for model reuse, complementary to feature extraction, is fine-tuning. Fine-tuning consists in unfreezing a few of the top layers of a frozen model base used for feature extraction, and jointly training both the newly added part of the model (in our case, the fully-connected classifier) and these top layers. This is called "fine-tuning" because it slightly adjusts the more abstract representations of the model being reused, in order to make them more relevant for the problem at hand.

fine_tuning.png

We have stated before that it was necessary to freeze the convolution base of MobileNet in order to be able to train a randomly initialized classifier on top. For the same reason, it is only possible to fine-tune the top layers of the convolutional base ONLY AFTER the classifier on top has already been trained. If the classified wasn't already trained, then the error signal propagating through the network during training would be too large, and the representations previously learned by the layers being fine-tuned would be destroyed. Thus the steps for fine-tuning a network are as follow:

1) Add your custom network on top of an already trained base network.
2) Freeze the base network.
3) Train the part you added.
4) Unfreeze some layers in the base network.
5) Jointly train both these layers and the part you added.
We have already completed the first 3 steps when doing feature extraction. Let's proceed with the 4th step: we will unfreeze our conv_base, and then freeze individual layers inside of it.

As a reminder, this is what our convolutional base looks like:
> In Keras, freezing a network is done by setting its trainable attribute to False.
`conv_base.trainable = False`

```
from keras.applications import MobileNet

conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3), alpha=0.5)

conv_base.summary()
```
