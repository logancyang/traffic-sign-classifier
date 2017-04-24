# **Traffic Sign Recognition**

Logan Yang
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image0]: ./examples/classes.png "Visualization"
[image1]: ./examples/class_distribution.png "Distribution"
[image2]: ./examples/normalization.png "Normalization"
[image3]: ./examples/grayscale.png "Grayscale"
[image4]: ./examples/random_noise0.png "Translation"
[image5]: ./examples/random_noise1.png "Rotation"
[image6]: ./examples/random_noise2.png "Zoom"
[image7]: ./examples/new_distribution.png "Augmented Data"

[image8]: ./my-signs/00000.png "Traffic Sign 1"
[image9]: ./my-signs/00003.png "Traffic Sign 2"
[image10]: ./my-signs/00004.png "Traffic Sign 3"
[image11]: ./my-signs/00005.png "Traffic Sign 4"
[image12]: ./my-signs/00007.png "Traffic Sign 5"
[image13]: ./my-signs/00010.png "Traffic Sign 6"
[image14]: ./examples/results.png "Traffic Sign 6"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is `32, 32, 3`
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First, all classes are shown with a representative image. Next, a bar chart showing how the classes are distributed. Some of the classes are over-represented and some are under-represented. The idea of data augmentation comes in at this point.

![alt text][image0]

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: If you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, the images need to be preprocessed. I normalized the images to `[0, 1]` and converted them to grayscale because the significance of color is negligible in this problem.

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

Next, given the imbalance training set, I generated images by translating, rotating or zooming the existing images by a random small amount. In this way, new images can be generated and the key information in the image, aka the sign is still preserved.

Here is an example of an original image and an augmented image:

![alt text][image4]
![alt text][image5]
![alt text][image6]

The difference in distribution between the original data set and the augmented data set is shown below

![alt text][image7]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer             |     Description                   |
|:---------------------:|:---------------------------------------------:|
| Input             | 32x32x1 grayscale image                 |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU          |                       |
| Max pooling         | 2x2 stride,  outputs 14x14x6         |
| Convolution 5x5     | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU          |                       |
| Max pooling         | 2x2 stride,  outputs 5x5x16         |
| Convolution 5x5     | 1x1 stride, valid padding, outputs 1x1x400    |
| RELU          |                       |
| Fully connected   | Flatten the two previous convnets 5x5x16 and 1x1x400, concatenate to size 800, outputs 400  |
| RELU          |                       |
| DROPOUT          | Dropout with probability 0.5                      |
| Fully connected   | outputs 43  |
| RELU          |                       |
| Softmax       | Return the logits  |
|           |                       |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used `AdamOptimizer` which is better than vanilla gradient descent.

```
EPOCHS = 25
BATCH_SIZE = 150
rate = 0.002
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%.
* validation set accuracy of 97.0%.
* test set accuracy of 94.4%.

An iterative approach was chosen:
* The first architecture was the vanilla LeNet5 from the lab. It yielded ~90% test accuracy out-of-the-box, which was not bad. However, both training and validation accuracies are not high, indicating there's underfitting. The architecture couldn't catch enough details from the dataset, so a deeper / more complex model is needed.
* The next starting point should be an architecture from the Yann LeCun paper on traffic sign classification. I adapted to its description, added an extra fully connected layer, then managed to get a validation accuracy of ~94%, a significant advantage over the previous model.
* By printing training accuracy, clearly there was overfitting happening - as the training accuracy went up, validation accuracy went down after it hit 94%. So DROPOUT is added, the validation accuracy went up to ~97%.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12] ![alt text][image13]

These images have different resolutions, I transformed them to 32 by 32 but the difference in size and image quality is still a factor that could affect the prediction performance.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image             |     Prediction                    |
|:---------------------:|:---------------------------------------------:|
| Vehicles over 3.5 metric tons prohibited         | Vehicles over 3.5 metric tons prohibited                     |
| Turn right ahead          | Turn right ahead                    |
| Right-of-way at the next intersection         | Right-of-way at the next intersection                     |
| Keep right            | Keep right                  |
| Ahead only     | Ahead only                   |
| No entry     | No entry                   |


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][image14]

