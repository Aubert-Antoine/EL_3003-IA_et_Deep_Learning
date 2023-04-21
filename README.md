# EL_3003-IA et Deep Learning 

Made in pairs with [Pauline Gouillart](https://github.com/pauline-gllrt)


## Unit description : 
The unit is an introdution to deep learning, with keras. We studied ConvNet and Dense neural netwoks. 
A focus is made on hyper-parameters, and regularisation techniques. The principal data proced are images. 
- MNIST dataset
- custom dataset

## Project : 
The final project is to make face rocognition on a small data base with a CNN, and then compare the efficency with other kind of classifier, such as KNN, SVM, and so on. 

### Goals : 
##### 1. Face detection : 
The first step in your pipeline is face detection. Normalize the cropped faces (i.e., divide the pixel values by 255), and split them in train set (70%) and test set (30%) with the function train_test_split in the package sklearn.Train a small convnet and check its performance on the test set. Then try to improve the performance of the baseline convnet by using all the tricks you have learned in the course.

##### 2. Pose estimation : 
You have isolated the faces in our image. But now you have to deal with the problem that faces turned different directions look totally different to a computer. To account for this, you will try to warp each picture so that the eyes and lips are always in the same place in the image. More concretely, you are going to use an algorithm called face landmark estimation. The basic idea is to locate 68 specific points (called landmarks) that exist on every face:  the top of the chin, the outside edge of each eye, the inner edge of each eyebrow, etc. Then, you’ll simply rotate, scale and shear the image so that the eyes and mouth are centered as best as possible. This will make face recognition more accurate.

##### 3. Face encoding : 
The training process works by looking at 3 face images at a time: the picture of a known person, another picture of the same known person, and a picture of a totally different person. Then, the algorithm looks at the measurements currently generated for each of those three images. It tweaks the neural network slightly to make sure that the measurements generated for the same person are slightly closer, and the measurements for different persons are slightly further apart. After repeating this step millions of times for millions of images of thousands of different people, the convnet learns to generate 128 measurements for each person.


##### 4. Face recognition : 
This last step is actually the easiest one in the whole process. All you have to do is find the person in your database of known people who has the closest measurements to some test image. You can do that by using any machine learning classification algorithm, such as neaural network (as you did in the previous section), logistic regression, SVM, nearest neighbours, etc. All you need to do is training a classifier that can take in the measurements from a new test image, and tells which known person is the closest match. Running this classifier must only take milliseconds, so that you can apply it to video sequences.


### Data set :
The data set is a small folder a pictures (207) and the linked label. The images are not cropend, not normalize and nor aligne. 
We process to do so and then encode faces. At each steps we run convnet to compare the results. 

### Packages :
- cv2 
- dlib
- tensorflow / keras
- numpy 
- mathplotlib
- scikit-learn

### Résults : 
...
