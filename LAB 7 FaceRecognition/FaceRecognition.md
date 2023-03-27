# Face recognition Final Project : 

Credit
https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78

CR Template
https://laurentnajman.org/uploads/images/DeepLearning/IEEE_Report_Template.pdf


# Architechture : 

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

**Type of *loss***
1. loss='binary_crossentropy'
    `model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`

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
