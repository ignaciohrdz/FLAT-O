# FLAT-O: Facial Landmark Annotation Tool with OpenCV

![GUI screenshot](docs/screenshot.png)

_(Photo by Thgusstavo Santana from [Pexels](https://www.pexels.com/photo/man-with-cigarette-in-mouth-1933873/))_

## 🎯 Features

- Simple OpenCV GUI for **68-keypoint** facial landmark annotation.
- Annotate **more than one face per image**.
- Annotation modes:
  - **Freehand**: Draw the curve for the face part and the landmarks will be interpolated.
  - **Polyline**: Mark as many points as you like for a face part and the final landmarks will be interpolated.
  - **Point**: If you don't want any fitting or interpolation at all, you cant just use the points you selected as the final annotations for the face part.
- Export the annotations in a single XML file, following the [dlib's example XML file](https://github.com/davisking/dlib/blob/master/examples/faces/training_with_face_landmarks.xml).

The goal behind freehand and polyline modes is to make your life easier. For each feature (nose, eyes, eyebrows...) mark as many points as you like and the final **keypoints will be calculated automatically so that they are distributed evenly**. This means that you are not limited to a certain number of points per feature. You use as many as you need and the program will extract the correct number of keypoints for that feature.

I have implemented two interpolation methods:

1. **Pick _N_ equidistant points** along the path defined by the points you selected.
2. **Fit a curve to your selected points** and then pick _N_ equidistant points along that curve (according to the number of points of the current face part). The goal of this second method is to provide softer curves, but it may not work sometimes.

### ✏ 68 facial landmark annotation

Here is the template for annotating the 68 keypoints of a face:

![Annotation of 68 facial landmarks](docs/68_landmarks.jpg)

### How to annotate with FLAT-O

📦 You will need the following packages:

- OpenCV
- Numpy
- Pandas
- Scipy

Once you have them installed, you just need to run this command:

    python annotate.py -i path\to\your\images -x your\xml\folder

📂 Make sure that all your images are in the same folder (`path\to\your\images`). By default, the program will split your data into training and test (10%). If you add the argument `--no-splits` it will only create one XML. The argument `-t` adjusts the test set size. This example sets the test set size to the 20%:

    python annotate.py -i path\to\your\images -x your\xml\folder -t 20

Finally, you can change the display size with the argument `-d`. This example scales the images so that the largest dimension is 512 pixels long (but it won't affect the original images):

    python annotate.py -i path\to\your\images -x your\xml\folder -t 20 -d 512

Remember that in order to train a shape predictor `dlib` requires the XML files (train and test) to be in the same folder as the images.

#### ⚠ How to annotate eyes and lips

Eyes and lips are the only features that are closed shapes, and that would break the method I'm using for automatically spacing the keypoints. For that reason, I have separated the annotation of eyes and lips in many parts:

- 👁 Eyes
    - Upper eyelid:
        - Left: 37 to 40
        - Right: 43 to 46
    - Lower eyelid:
        - Left: [40, 41, 42, 37]
        - Right: [46, 47, 48, 43]
- 👄 Mouth
    - Outer
        - Upper lip: 49 to 55
        - Lower lip: [55, 56, 57, 58, 59, 60, 49]
    - Inner
        - Upper lip: 61 to 65
        - Lower lip: [65, 66, 67, 68, 61]

This means that, in order to annotate the lower eyelids and lips properly, you will have to mark the edge points (i.e. 37, 40, 43, 46, 49, 55, 61, 65) twice.

##### Example

Let's suppose you are annotating the left eye:

1. First, you mark the keypoints for the upper eyelid: 37, 38, 39 and 40.
2. You press `<space>` and the final keypoints are generated.
3. You move onto the next feature (lower eyelid) and mark the keypoints 40, 41, 42 and 37. However, when you press `<space>` again the keypoints 37 and 40 won't be overwritten because they were already saved in the previous part.

## ⌨ Keyboard controls

- `<z>`/`<x>`: Previous/next face part
- `<a>`/`<s>`: Previous/next face
- `<space>`: If you press it after clicking some points, it will generate the corresponding keypoints for the current face part. Every time you press `<space>` the program will jump onto the next part.
- `<f>`: Set curve fitting on/off. In case you use curve fitting and get `RankWarning: Polyfit may be poorly conditioned`, you are probably using too few points (try always to use at least 3).
- `<m>`: Change annotation mode between Freehand, Polyline and Point.
- `<u>`: Undo annotation of the current face part
- `<r>`: Reset the entire face (only applies to the face you are currently annotating)
- `<q>`: Quit. Use this command to exit the program and save the annotations. If you just close the window, the annotations won't be saved.

## Useful links

Other annotation tools:

- [Facial Landkmarks Annotation Tool (FLAT)](https://github.com/luigivieira/Facial-Landmarks-Annotation-Tool)
- [imglab (dlib's annotation tool)](https://github.com/davisking/dlib/tree/master/tools/imglab)

Learn about `dlib` here:

- [Training a custom dlib shape predictor](https://pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/)
- [Facial landmarks with dlib, OpenCV and Python](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
- [Training script example from dlib](http://dlib.net/train_shape_predictor.py.html)
- [Facial Landmarks: Use Cases, Datasets, and a Quick Tutorial](https://datagen.tech/guides/face-recognition/facial-landmarks/)
- [Face Landmark Detection using Dlib](https://debuggercafe.com/face-landmark-detection-using-dlib/)