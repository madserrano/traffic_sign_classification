# Traffic_sign_classification
Traffic Sign Classification model using CNN

Image classification is a Machine Learning method that recognizes an input image and predicts its category or class. Several factors and challenges are associated with building classification models, such as image variations, imbalanced datasets, and overfitting. The most common network model for building an image classification problem is Convolutional Neural Network (CNN). Tensorflow Keras and Pytorch are the most widely used libraries to train and compile a CNN model. Training a CNN model requires just enough features of an image to produce a good result, unlike other types of Neural Networks. In this project, CNN was leveraged to solve a Traffic Sign Classification problem.

## Dataset:
The data used in this project is the GTSRB (German Traffic Sign Recognition Benchmark) dataset which originated from INI Benchmark website. It was a collection of real-life images of German traffic road signs and was formerly used for a competition at IJCNN in 2011.
A total of 3,3799 images were assigned for training data, labelled with 43 classes. Test data consist of 12,630 non-labelled images, and 4,410 images were used for validation.

## Method:
Three methods were conducted to generate a model that could provide the most logical result. Two methods used Keras sequential API and the other one used Pytorch. As a result of this experiment, Method 2, which leveraged LeNet architecture, produced the highest validation score of 94%. Evaluation using new image sets of German traffic signs yield 60-70%, while non-German traffic signs produced poor accuracy. 

It was concluded that a good quality model can be achieved with adequate and balanced dataset. The accuracy of Model 2 can be sufficient for experimental level, but for a more robust model, one must invest a great amount of time and sufficient machine processor to train it.

The final version of this project runs in a public DNS with URL: http://ts-classification.herokuapp.com/. Streamlit web framework was used to build the user interface and the whole set-up was compiled in a docker image, deployed to Heroku cloud service.


