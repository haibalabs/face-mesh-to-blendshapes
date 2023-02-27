Copyright (c) 2023 Haiba Labs

Author: James Ritts <james@haibalabs.com>

This notebook trains a simple pytorch model to map from [MediaPipe face mesh](http://solutions.mediapipe.dev/face_mesh) landmarks to [ARKit-compatible blendshapes](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation).

### [Click here to open the demo.](https://haibalabs.github.io/face-mesh-to-blendshapes/test/mediapipe_to_arkit.html)

Caveats
-
- We wish to train on object space geo so it doesn't have to learn what a face pose looks like in every possible head orientation. Unfortunately MediaPipe's output is only given in [screen coordinates](https://www.cse.iitd.ac.in/~suban/vision/affine/node5.html). Its mesh is also stretched to conform to the silhouette of the face in the input image. The function normalize_landmarks() tries to undo these effects.
- The function convert_landmarks_to_model_input() uses normalize_landmarks in order to convert from raw MediaPipe output to the NN input vector. This function needs to be ported to any environment where the model is run.
- MediaPipe isn't able to signal every blendshape. These should be forced to zero at runtime and possibly others as well: jawForward, jawRight, jawLeft, mouthDimpleRight, mouthDimpleLeft, cheekPuff, tongueOut.


Format
-
The order of blendshape values in the model output is:

```
eyeBlinkRight, eyeLookDownRight, eyeLookInRight, eyeLookOutRight, eyeLookUpRight, eyeSquintRight, eyeWideRight, eyeBlinkLeft, eyeLookDownLeft, eyeLookInLeft, eyeLookOutLeft, eyeLookUpLeft, eyeSquintLeft, eyeWideLeft, jawForward, jawRight, jawLeft, jawOpen, mouthClose, mouthFunnel, mouthPucker, mouthRight, mouthLeft, mouthSmileRight, mouthSmileLeft, mouthFrownRight, mouthFrownLeft, mouthDimpleRight, mouthDimpleLeft, mouthStretchRight, mouthStretchLeft, mouthRollLower, mouthRollUpper, mouthShrugLower, mouthShrugUpper, mouthPressRight, mouthPressLeft, mouthLowerDownRight, mouthLowerDownLeft, mouthUpperUpRight, mouthUpperUpLeft, browDownRight, browDownLeft, browInnerUp, browOuterUpRight, browOuterUpLeft, cheekPuff, cheekSquintRight, cheekSquintLeft, noseSneerRight, noseSneerLeft, tongueOut
```

Training data has this folder structure:
- my_first_dataset
  - neutral.jpg
  - my_first_dataset.csv
  - my_first_dataset_000000.jpg
  - my_first_dataset_000001.jpg
  - my_first_dataset_000002.jpg
  - ...
- my_second_dataset
- sets.txt

The file sets.txt should contain the folder names of all training datasets:
```
my_first_dataset
my_second_dataset
...
```

Each dataset must have a calibration photo depicting a neutral facial expression: **neutral.jpg**.  The model is trained on object space offsets from the neutral pose.

Each dataset also has a CSV file containing a header row followed by labels (blendshape values) for each input image:
```
eyeBlinkRight,eyeLookDownRight,eyeLookInRight,eyeLookOutRight,eyeLookUpRight,eyeSquintRight,eyeWideRight,eyeBlinkLeft,eyeLookDownLeft,eyeLookInLeft,eyeLookOutLeft,eyeLookUpLeft,eyeSquintLeft,eyeWideLeft,jawForward,jawRight,jawLeft,jawOpen,mouthClose,mouthFunnel,mouthPucker,mouthRight,mouthLeft,mouthSmileRight,mouthSmileLeft,mouthFrownRight,mouthFrownLeft,mouthDimpleRight,mouthDimpleLeft,mouthStretchRight,mouthStretchLeft,mouthRollLower,mouthRollUpper,mouthShrugLower,mouthShrugUpper,mouthPressRight,mouthPressLeft,mouthLowerDownRight,mouthLowerDownLeft,mouthUpperUpRight,mouthUpperUpLeft,browDownRight,browDownLeft,browInnerUp,browOuterUpRight,browOuterUpLeft,cheekPuff,cheekSquintRight,cheekSquintLeft,noseSneerRight,noseSneerLeft,tongueOut
0.039,0.103,0.044,0.000,0.000,0.000,0.000,0.039,0.104,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.010,0.010,0.027,0.000,0.000,0.002,0.003,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.015,0.014,0.000,0.000,0.000,0.007,0.000,0.000,0.000,0.000,0.000
0.038,0.091,0.049,0.000,0.000,0.000,0.000,0.038,0.092,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.010,0.011,0.027,0.000,0.000,0.002,0.004,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.014,0.014,0.000,0.000,0.000,0.007,0.000,0.000,0.000,0.000,0.000
...
```

Relevant links:
-
- https://arxiv.org/pdf/2006.10962.pdf
- https://developers.googleblog.com/2020/09/mediapipe-3d-face-transform.html
- https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_geometry/data
- https://github.com/google/mediapipe/issues/2867
- https://stackoverflow.com/questions/69858216/mediapipe-facemesh-vertices-mapping
- https://github.com/Rassibassi/mediapipeFacegeometryPython/blob/main/face_geometry.py
- https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png


To do:
-
- find ways to reduce head rotation-driven blendshape error
- cull shapes from NN output and training with which MP points don't correlate
- cull training examples which don't signal shapes that MP can detect
