# Image-Features-Detection--OpenCV-Python

<img src="images/banner-01.png" width="1000"/>

## 1. Objective

The objective of this section is to demonstrate the detection of various types of image features, using OpenCV Python. In particular, we shall detect and visualize the following features:

  * Harris corners
  * SURF
  * BRIEF
  * ORB.



## 2. Motivation

A feature in human vision is a region of interest in an image that is unique and easy to recognize. Features include things like, points, edges, blobs, and corners. Thus, features are distinguishable image pixel or region based characteristics that are associated with certain identifiable objects or scenes.  Thus, in order to recognize an object or scene, our human visual system first scans the scene, our brain detects the key points or features and then recognizes the scene by matching or associating the detected features with known objects or scene stored in our memory. Thus, the feature detection is a critical first step in object and scene recognition.

A computer vision systems follow a similar object and scene recognition process:

  * Acquire the image of the scene
  * Detect key points and features
  * Try to match the detected features with those extracted from known objects stored in the data base
  * Identify the new imaged object in the scene based on any successful matches in step 3:
  * The data base object yielding a sufficiently high number of matches, if there is one.

Image features detection algorithms perform two steps:

* Feature extraction: 
  * A feature detector finds regions of interest in an image. 
  * The input into a feature detector is an image, and the output are pixel coordinates of the significant areas in the image.
* Feature description:
  * A feature descriptor encodes that feature into a numerical “fingerprint”. 
  * Feature description makes a feature uniquely identifiable from other features in the image. 
  * We can then use the numerical fingerprint to identify the feature even if the image undergoes some type of distortion.

In this section, we shall review, implement and illustrate a selected subset of image feature algorithms that are available in OpenCV.

## 3. Data

The in put image used to illustrate the feature detection OpenCV functionalities is illustrated in the figure below.

<img src="images/input-image.jpg" width="1000"/>

## 4. Development

In this section, we shall demonstrate detecting and visualizing the image features, mentioned earlier, using OpenCV Python

* Author: Mohsen Ghazel (mghazel)
* Date: March 29th, 2021
* Project: Image Features Detection

The objective of this project is to demonstrate how to detect various types of image features using OpenCV with Python API, which include the following features:

  * Harris corners
  * FAST
  * BRIEF
  * ORB

We shall assess these various types of features in terms of the type of information each feature captures or extracts from the image


### 4.1. Step 1: Python imports:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># system environment</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># Numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#696969; "># image processing library</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>image <span style="color:#800000; font-weight:bold; ">as</span> mpimg
<span style="color:#696969; "># date and time</span>
<span style="color:#800000; font-weight:bold; ">import</span> datetime

<span style="color:#696969; "># check for successful package imports and versions</span>
<span style="color:#696969; "># python</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># numpy</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Python version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.7</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">10</span> <span style="color:#808030; ">(</span>default<span style="color:#808030; ">,</span> Feb <span style="color:#008c00; ">20</span> <span style="color:#008c00; ">2021</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">17</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">23</span><span style="color:#808030; ">)</span> 
<span style="color:#808030; ">[</span>GCC <span style="color:#008000; ">7.5</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span> 
OpenCV version <span style="color:#808030; ">:</span> <span style="color:#008000; ">4.1</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span> 
Numpy version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">5</span> 
</pre>


<pre style="color:#000000;background:#ffffff;"><span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#074726; ">__doc__</span><span style="color:#808030; ">)</span>

Automatically created module <span style="color:#800000; font-weight:bold; ">for</span> IPython interactive environment
</pre>

### 4.2. Step 2: Read and visualize the input image


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># The input image file name</span>
img_file_path_name <span style="color:#808030; ">=</span> os<span style="color:#808030; ">.</span>path<span style="color:#808030; ">.</span>join<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"sample_data"</span><span style="color:#808030; ">,</span><span style="color:#0000e6; ">"chessboard-scene.jpg"</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># read the input image</span>
img <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>imread<span style="color:#808030; ">(</span>img_file_path_name<span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if the image is read successfully</span>
<span style="color:#800000; font-weight:bold; ">if</span> img <span style="color:#800000; font-weight:bold; ">is</span> <span style="color:#074726; ">None</span><span style="color:#808030; ">:</span>
    sys<span style="color:#808030; ">.</span>exit<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Could not read the image file: "</span> <span style="color:#44aadd; ">+</span> img_file_path_name<span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if it is grayscale image, if so convert it to RGB by </span>
<span style="color:#696969; "># duplicating the channel</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  img <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>merge<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span>img<span style="color:#808030; ">,</span>img<span style="color:#808030; ">,</span>img<span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># check if it is color image, if so convert it to grayscale</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    gray <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>cvtColor<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>COLOR_BGR2GRAY<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> <span style="color:#696969; "># make a copy of the image</span>
    gray <span style="color:#808030; ">=</span> img<span style="color:#808030; ">.</span>copy<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># display the input image </span>
<span style="color:#696969; "># create a figure</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Input image: Chessboard scene"</span><span style="color:#808030; ">,</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the original image</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">111</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Input image: Chessboard scene"</span><span style="color:#808030; ">,</span> fontsize <span style="color:#808030; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the original image</span>
<span style="color:#696969; "># - if the image is RGB</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span><span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> <span style="color:#696969; "># for grayscale image</span>
    plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/input-image.jpg" width="1000"/>

### 4.3. Step 3: Harris corners:

* Harris Corner Detector extracts corner-like features from the image:
  * A corner is typically located at the intersection of 2 edge-lines.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#======================================</span>
<span style="color:#696969; "># 3) Harris-Corners detection</span>
<span style="color:#696969; ">#======================================</span>
<span style="color:#696969; "># make a copy of the image</span>
copy <span style="color:#808030; ">=</span> img<span style="color:#808030; ">.</span>copy<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>

<span style="color:#696969; "># convert the image to float32</span>
gray <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>float32<span style="color:#808030; ">(</span>gray<span style="color:#808030; ">)</span>

<span style="color:#696969; "># apply the Harris corner</span>
dst <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>cornerHarris<span style="color:#808030; ">(</span>gray<span style="color:#808030; ">,</span><span style="color:#008c00; ">2</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span><span style="color:#008000; ">0.04</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># dilate the initial detections to remove unimportant corners</span>
dst <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>dilate<span style="color:#808030; ">(</span>dst<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># Threshold the detected corners:</span>
<span style="color:#696969; "># - the optimal threshold value, it may vary depending on the image.</span>
yy<span style="color:#808030; ">,</span>xx <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>where<span style="color:#808030; ">(</span>dst<span style="color:#44aadd; ">&gt;</span><span style="color:#008000; ">0.01</span><span style="color:#44aadd; ">*</span>dst<span style="color:#808030; ">.</span><span style="color:#400000; ">max</span><span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># visualize the final corner detections </span>
<span style="color:#800000; font-weight:bold; ">for</span> counter <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>xx<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    <span style="color:#696969; "># get the x-coordinate</span>
    x <span style="color:#808030; ">=</span> xx<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
    <span style="color:#696969; "># get the x-coordinate</span>
    y <span style="color:#808030; ">=</span> yy<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
    <span style="color:#696969; "># draw a BLUE circle at the point (x,y)</span>
    cv2<span style="color:#808030; ">.</span>circle<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span><span style="color:#808030; ">(</span>x<span style="color:#808030; ">,</span>y<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">,</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># visualize the figure</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">111</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># figure title</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Harris-Shi-Tomasi Corner Detection - OpenCV"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># axis off</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># diplay the image with overlays</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/Harris-corners.jpg" width="1000"/>

### 4.4. Step 4: FAST features detection

* FAST (Features from Accelerated Segment Test) algorithm
* It is several times faster than other existing corner detectors 
* It is more suitable for real-time applications
* But it is generally not robust to high levels of noise.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#======================================</span>
<span style="color:#696969; "># 4) FAST - Features from Accelerated Segment Test</span>
<span style="color:#696969; ">#======================================</span>
<span style="color:#696969; "># make a copy of the image</span>
copy <span style="color:#808030; ">=</span> img<span style="color:#808030; ">.</span>copy<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>

<span style="color:#696969; "># Initiate FAST object with default values</span>
fast <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>FastFeatureDetector_create<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># comute the keypoints</span>
kp <span style="color:#808030; ">=</span> fast<span style="color:#808030; ">.</span>detect<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># draw the keypoints</span>
img2 <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>drawKeypoints<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span> kp<span style="color:#808030; ">,</span> <span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> flags<span style="color:#808030; ">=</span>cv2<span style="color:#808030; ">.</span>DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS<span style="color:#808030; ">)</span>

<span style="color:#696969; "># Print default params</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Threshold: "</span><span style="color:#808030; ">,</span> fast<span style="color:#808030; ">.</span>getThreshold<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"nonmaxSuppression: "</span><span style="color:#808030; ">,</span> fast<span style="color:#808030; ">.</span>getNonmaxSuppression<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"neighborhood: "</span><span style="color:#808030; ">,</span> fast<span style="color:#808030; ">.</span>getType<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Total Keypoints with nonmaxSuppression: "</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>kp<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># Disable nonmaxSuppression</span>
fast<span style="color:#808030; ">.</span>setNonmaxSuppression<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># recompute the keypoints</span>
kp <span style="color:#808030; ">=</span> fast<span style="color:#808030; ">.</span>detect<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Total Keypoints without nonmaxSuppression: "</span><span style="color:#808030; ">,</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>kp<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># overlay the results without nonmaxSuppression</span>
img3 <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>drawKeypoints<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span> kp<span style="color:#808030; ">,</span> <span style="color:#074726; ">None</span><span style="color:#808030; ">,</span> flags<span style="color:#808030; ">=</span>cv2<span style="color:#808030; ">.</span>DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS<span style="color:#808030; ">)</span>

<span style="color:#696969; "># visualize the results</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">18</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># FAST Features with Non-Max Supression</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">121</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"FAST Features: WITH Non-Max Supression"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img2<span style="color:#808030; ">)</span>
<span style="color:#696969; "># FAST Features without Non-Max Supression</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">122</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"FAST Features: WITHOUT Non-Max Supression"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>img3<span style="color:#808030; ">)</span>

Threshold<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">10</span>
nonmaxSuppression<span style="color:#808030; ">:</span>  <span style="color:#074726; ">True</span>
neighborhood<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">2</span>
Total Keypoints <span style="color:#800000; font-weight:bold; ">with</span> nonmaxSuppression<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">10964</span>
Total Keypoints without nonmaxSuppression<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">29157</span> 
</pre>


<img src="images/FAST-features.jpg" width="1000"/>

### 4.5. Step 5: BRIEF features

* BRIEF is a faster method feature descriptor calculation and matching.
* It also provides high recognition rate unless there is large in-plane rotation.
* One important point is that BRIEF is a feature descriptor, it doesn’t provide any method to find the features. So you will have to use any other feature detectors like SIFT, SURF etc.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#======================================</span>
<span style="color:#696969; "># 5) BRIEF - Binary Robust Independent Elementary Features</span>
<span style="color:#696969; ">#======================================</span>
<span style="color:#696969; "># make a copy of the image</span>
copy <span style="color:#808030; ">=</span> img<span style="color:#808030; ">.</span>copy<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># Initiate BRIEF detector</span>
star <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>xfeatures2d<span style="color:#808030; ">.</span>StarDetector_create<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># Initiate BRIEF extractor</span>
brief <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>xfeatures2d<span style="color:#808030; ">.</span>BriefDescriptorExtractor_create<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># find the keypoints with STAR</span>
kp <span style="color:#808030; ">=</span> star<span style="color:#808030; ">.</span>detect<span style="color:#808030; ">(</span>copy <span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># compute the descriptors with BRIEF</span>
kp<span style="color:#808030; ">,</span> des <span style="color:#808030; ">=</span> brief<span style="color:#808030; ">.</span>compute<span style="color:#808030; ">(</span>copy <span style="color:#808030; ">,</span> kp<span style="color:#808030; ">)</span>

<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Descriptor Size : "</span><span style="color:#808030; ">,</span> brief<span style="color:#808030; ">.</span>descriptorSize<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Descriptor Shape : "</span><span style="color:#808030; ">,</span> des<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>

cv2<span style="color:#808030; ">.</span>drawKeypoints<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span>kp<span style="color:#808030; ">,</span>copy<span style="color:#808030; ">,</span>flags<span style="color:#808030; ">=</span>cv2<span style="color:#808030; ">.</span>DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS<span style="color:#808030; ">)</span>

plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize <span style="color:#808030; ">=</span> <span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">111</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"BRIEF Features"</span><span style="color:#808030; ">,</span> fontsize <span style="color:#808030; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>


Descriptor Size <span style="color:#808030; ">:</span>  <span style="color:#008c00; ">32</span>
Descriptor Shape <span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">336</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">32</span><span style="color:#808030; ">)</span> 
</pre>

<img src="images/BRIEF-features.jpg" width="1000"/>


### 4.6. Step 6: ORB features

* ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance.
* First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them.
* It also use pyramid to produce multiscale-features.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#======================================</span>
<span style="color:#696969; "># 6) ORB - Oriented FAST and Rotated BRIEF</span>
<span style="color:#696969; ">#======================================</span>
<span style="color:#696969; "># make a copy of the image</span>
copy <span style="color:#808030; ">=</span> img<span style="color:#808030; ">.</span>copy<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
<span style="color:#696969; "># Initiate ORB detector</span>
orb <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>ORB_create<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># find the keypoints with ORB</span>
kp <span style="color:#808030; ">=</span> orb<span style="color:#808030; ">.</span>detect<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span><span style="color:#074726; ">None</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># compute the descriptors with ORB</span>
kp<span style="color:#808030; ">,</span> des <span style="color:#808030; ">=</span> orb<span style="color:#808030; ">.</span>compute<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span> kp<span style="color:#808030; ">)</span>

<span style="color:#696969; "># draw only keypoints location,not size and orientation</span>
cv2<span style="color:#808030; ">.</span>drawKeypoints<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">,</span> kp<span style="color:#808030; ">,</span> copy<span style="color:#808030; ">,</span> flags<span style="color:#808030; ">=</span>cv2<span style="color:#808030; ">.</span>DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS<span style="color:#808030; ">)</span>

plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span>figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">111</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"ORB Features"</span><span style="color:#808030; ">,</span> fontsize <span style="color:#808030; ">=</span> <span style="color:#008c00; ">12</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>xticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> plt<span style="color:#808030; ">.</span>yticks<span style="color:#808030; ">(</span><span style="color:#808030; ">[</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>copy<span style="color:#808030; ">)</span><span style="color:#808030; ">;</span>
</pre>

<img src="images/ORB-features.jpg" width="1000"/>

### 4.7. Step 7: End of Execution

* Display a successful end of execution message


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>


Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">03</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">31</span> <span style="color:#008c00; ">00</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">02</span><span style="color:#808030; ">:</span><span style="color:#008000; ">54.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye! 
</pre>

## 5. Analysis

* In view of the illustrated results, we make the following observations:

  * The different image features capture different image unique characteristics and contents.
  * The number of distribution of the different features vary widely
  * FAST appears to yield an excessive number of  image features. 
  * Since FAST is a fast algorithm for detecting corners in an image, its results are similar to the Harris corner detector, except much more dense.


## 6. Future Work

* We propose to explore the following tasks:
  * The following 2 feature detectors have been shown to perform very well:
  * Scale-Invariant Feature Transform (SIFT)
  * Speed-up robust features (SURF)
  * However, they have been removed from the latest open-source versions of OpenCV due to licensing issues:
  * We plan to explore other libraries with these important feature detectors.


## 7. References

1. OpenCV. Feature Detection and Description. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
2. OpenCV. Feature Detection. https://docs.opencv.org/3.4/d7/d66/tutorial_feature_detection.html
3. OpenCV. Introduction to SIFT (Scale-Invariant Feature Transform). https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
4. Automatic Addison. Image Feature Detection, Description, and Matching in OpenCV. https://automaticaddison.com/image-feature-detection-description-and-matching-in-opencv/#Scale-Invariant_Feature_Transform_SIFT
5. Analytics Vidha. A Detailed Guide to the Powerful SIFT Technique for Image Matching (with Python code). https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/



