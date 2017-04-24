**Vehicle Detection Project**
Alistair Kirk - April 24 2017

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

** NOTE: I have elected to use a Neural Network Approach instead of HOG/SVM's. This was pre-approved by my course mentory Aaron Smith.**

[//]: # (Image References)
[pandas]: ./output_images/pandasframes.png
[pandasbbox]: ./output_images/pandasbbox.png
[pandasmask]: ./output_images/pandasmask.png
[unetarch]: ./output_images/u-net-architecture.png
[predictions]: ./output_images/prediction_ground_truth.png
[image1]: ./output_images/videoframe1.png
[image2]: ./output_images/videoframe2.png
[image3]: ./output_images/videoframe3.png
[image4]: ./output_images/videoframe4-false-pos.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Because a different approache was chosen, the rubric will not directly apply for this project.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Assembling Training Data

#### Explain how (and identify where in your code) how the training images were assembled.

The code for this step is contained in the second code cell of the IPython notebook. 

I started by downloading the [Udacity Annotated Driving Set](https://github.com/udacity/self-driving-car/tree/master/annotations) that was derived from a [crowd-ai](https://crowdai.com/) exercise to obtain labelled training data for trucks, cars, and pedestrians.

The driving set contains ~4.5GB of images and a csv file that describes attributes of each image. Using pandas I created a dataframe of the vehicle list and removed extraneous information:
![pandas][pandas]

In the third code block of the notebook I created functions to obtain the bounding box coordinates from the DataFrame for all trucks and cars, and their corresponding images, and then plotted the bounding boxes to ensure that the correct information was obtained:

![pandasbbox][pandasbbox]

In the fourth code block, I created a function to generate a 1 channel mask image from the bounding boxes and initial image:

![pandasmask][pandasmask]

In the fifth code block, I collected images and masks into two lists in memory. I did this for convenience when starting the project and it lead to some challenges later on with memory space when trying to train the whole dataset. I had to eventually procedurally train the network in manageable blocks of approx. 4000 images, updating the weights. I learned an important lesson here about considering the dynamic allocation of large datasets from a hard drive. This complication was compounded by the fact that I had to adjust the size of my neural network to run on my GTX 770 GPU with 2GB of GPU memory limiting the size of the input image.

To deal with the GPU memory limitation, I decided to write a function that pre-processes the dashcam input video by focusing on the lower third of the image, and cropping out the sky and hood, then splitting the image into 2 left and right images. These images were also scaled down by 50%. This effectively transforms an input image with pixel dimensions of (1920,1200) down to (400, 480). This is a 12x reduction in the input image size, allownig my GPU to process the images.

In the sixth code block I split the images (`X_`) and masks (`y_`) into training and testing datasets, reserving 20%.

#### Explain the selection of the Neural Network

I chose the so-called U-Net for this vehicle recognition project, with credit for the theory and basis of the NN given to [Jocic Marko](https://github.com/jocicmarko/ultrasound-nerve-segmentation.git) which won the Kaggle competition for nerve segmentation, and [Vivek Yadav](https://chatbotslife.com/@vivek.yadav) who described his experience using the same network.

The U-Net is a segmentation type neural network, where the output of the first few convolutional layers are not only fed to their next conv layer, but also concatenated to the later convolutional layers which allows connected sharing of the information between the lower and upper layers and results in a fairly reliable image detection network.  I adopted the optimization proposed by Yadav which reduces layer sizes by half compared to that described by Marko. This U shaped architecture can be visualized as shown from Marko:

![unetarch][unetarch]

I also updated the network achitecture to use Keras 2, and trained it to use the reduced image input size (480,400):

_________________________________________________________________
Layer (type)                 Output Shape              Param #   

input_1 (InputLayer)         (None, 400, 480, 3)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 400, 480, 8)       224       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 400, 480, 8)       584       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 200, 240, 8)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 200, 240, 16)      1168      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 200, 240, 16)      2320      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 100, 120, 16)      0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 100, 120, 32)      4640      
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 100, 120, 32)      9248      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 50, 60, 32)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 50, 60, 64)        18496     
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 50, 60, 64)        36928     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 25, 30, 64)        0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 25, 30, 128)       73856     
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 25, 30, 128)       147584    
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 50, 60, 128)       0         
_________________________________________________________________
concatenate_1 (Concatenate)  (None, 50, 60, 192)       0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 50, 60, 64)        110656    
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 50, 60, 64)        36928     
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 100, 120, 64)      0         
_________________________________________________________________
concatenate_2 (Concatenate)  (None, 100, 120, 96)      0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 100, 120, 32)      27680     
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 100, 120, 32)      9248      
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 200, 240, 32)      0         
_________________________________________________________________
concatenate_3 (Concatenate)  (None, 200, 240, 48)      0         
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 200, 240, 16)      6928      
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 200, 240, 16)      2320      
_________________________________________________________________
up_sampling2d_4 (UpSampling2 (None, 400, 480, 16)      0         
_________________________________________________________________
concatenate_4 (Concatenate)  (None, 400, 480, 24)      0         
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 400, 480, 8)       1736      
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 400, 480, 8)       584       
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 400, 480, 1)       9         

Total params: 491,137.0
Trainable params: 491,137.0
Non-trainable params: 0.0
_________________________________________________________________

The loss function that is used in the NN is a modification of the so-called [Dice-Sorensen Coefficienct](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and explained in detail by Marko and Vivek. Essentially the loss function finds the intersection of the predicted mask and ground truth masks, and divides by the sum of the magnitude of each, and adds a smoothing scalar of 1, I believe to avoid cases of dividing by zero in case the predicted image or mask is zero:

```
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
```

#### Explain the Training Process of the Neural Network

The NN was trained progressively using ~4000 training images (less the 20% reserve for validation) at a time due to the system memory limitations imposed as described above. The network was trained for 10 epochs each time, and the next batch of training images was selected, and run again for 10 epochs, until all available images in the database were used. Future work would consider a dynamic image and mask generator, this was complicated by the fact that the input layer images need to be cropped and scaled from the training data. An intermediate fix would be to generate the training images and save them to disk.

Using the Keras generator and fit functions as shown in the 12th code block, the training images (as X values) and masks (as y values) are combined in an interesting way according to the [Keras documentation](https://keras.io/preprocessing/image/), that allows the masks to be used as the 'labels' for training the network. Using this approach, the automatic data augmentation features of Keras are available and were used to augment the training data by rotating +- 15 degrees (rotation range), and translating the images horizontally and vertically by 10% (width/height shift range), and scaling the images by 20% (zoom range).

A Model Checkpoint was used to update the weights only when the loss functions were improved for each epoch.

In total the model was run for over 50 epochs on all the training data, yielding a final loss value of ~ 0.71 or 71% which roughly correlates with the success of the Kaggle competition. This could indicate that the Neural Network is limited to being approximately 71% correct unless the architecture is modified or improved and remains as a potential future improvement.

This level of accuracy turned out to be good enough for the vehicle detection project, once further measures are taken to smooth the data as described in the later sections.

#### Testing the Network

As shown in the 15th code block of the notebook, a selection of images from the Udacity data set were then fed into the Neural Network to make prediction, and the final predicted masks, shown in red, look very promising when compared to the green 'ground-truth' masks:

![predictions][predictions]

#### Pipeline Production and Smoothing Parameters

As shown in the 16th code block, a Car Class was created to store essential information for each vehicle that is detected including the `n` most recent bounding boxes, a simple counter `self.detected` that is used in determining how long the car has been detected (in frames), and if the car is consistently detected in each new frame, this counter increases at a rate of 1.5, however if the car has not been detected the counter decays at a rate of 1.0 per frame, until it is essentially 'forgotten' and removed from the list of Cars.

In the 17th code block, I set up a production pipeline of functions that:
* Use CLAHE to correct image luminosity
* Get the predicted mask or 'heat map' image for each frame
* Use scipy's label function to generate a labelled image and unique count of hotspots from the predicted mask
* Get the boundary boxes for each unique labelled car

additional functions in here include:
* a distance calculation between cars - if they are within 150 pixels, the cars are considered part of the same car and are merged together
* draw boundary boxes - takes an input image and list of Car objects, then draws their `n` averaged boundary box dimensions, included in here are some sanity checks for whether any newly detected cars are part of the original Car List, and if they are they are merged, and also any boundary boxes that are smaller than 400 pixels in total are ignored, this reduced the number of false positives.

In the 18th code block the main image processing pipeline function was defined. This proved to be a bit of a challenge because the video input images were at a different resolution than the full size training images (1280, 960) compared to (1920,1200), so I had to scale the images up using CV2's `INTER_CUBIC` method, then slice and scale the images to the expected input layer size, and once the predicted masks were received, they were padded and upscaled to the original image size before processing in the label and boundary box functions, so that the same video resolution would be received once the detection was completed.

The parameters for smoothing (i.e. boundary box minimum areas, the vehicle detection counter, and the merging proximity of cars) were all tuned after several runs of the input video to provide a satisfactory balance between speed of vehicle recognition and minimization of false positives.

Here we see a few select frames from the video:

![image1][image1]
Here the NN detects the vehicle in the immedite foreground, but what is also interesting is that it picks up oncoming vehicle traffic, although most of the oncoming vehicle traffic is suppressed with the smoothing of false positives. I believe this feature is important to keep in a robust system that would be able to detect traffic that the vehicle is traveling with and oncoming, and should be able to further classify the types of traffic each vehicle is in, so that it can make better decisions on whether traffic is doing something expected or unexpected.

![image2][image2]
![image3][image3]
Here the NN is working as desired by applying bounding boxes around both cars individually.

![image4][image4]
Here the highest region of false positives are shown as the NN detects vehicles for more than a few frames in the shadows of the tree on the edge of the screen. Further refinement of the process could reduce these false positives by considering the momentum and vector of each detected vehicle and ignoring roadside detections, or by improving the NN to more reliably classify and detect vehicles, of which there are already some amazing improvements being made to date.

### Video Implementation

Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./my_test.mp4)

Overall the NN performs quite well after tuning some of the smoothing parameters to minimize false positive detection.

The video took ~2.25 minutes to process using the NN approach, which is significantly faster than the recommended HOG/SVM approach as reported by other students in the course.

---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The Neural Network works remarkably well for vehicle detection, and more importantly is quite fast at approximately 8 fps on my i7 desktop with GTX 770 GPU, and 16GB of system RAM. The speed of image processing could be further improved by optimizing the CarList sanity checks and smoothing functions, and by rethinking how the input images are scaled before and after the vehicle detection.

Due to the reduced resolution of the images in the input layer of the NN, vehicles in the farfield tend to not be reliably detected and can be see in the video. Only once the gray van gets past the minimum detection size (~400 pixels) the boundary box is applied, but it does not last very long. The accuracy of farfield vehicle recognition is minimized as a trade-off to reduce the number of false positives.

Future optimization would consider trying different NN architecture, possibly by increasing the layer sizes or experimenting with dropout and larger training data sets to improve vehicle prediction. Further refinement of the smoothing parameters could also be explored to reduce the number of false positives and ensure fast vehicle detection.

