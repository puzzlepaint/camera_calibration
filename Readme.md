## Accurate geometric camera calibration ##

* [Overview](#overview)
* [About](#about)
* [Building](#building)
* [How to use](#how-to-use)
  * [Obtaining a calibration pattern](#obtaining-a-calibration-pattern)
  * [Calibrating a camera with live input](#calibrating-a-camera-with-live-input)
  * [Calibrating a camera from images in a folder](#calibrating-a-camera-from-images-in-a-folder)
  * [Calibrating a stereo camera and computing depth images](#calibrating-a-stereo-camera-and-computing-depth-images)
  * [Which camera model to choose?](#which-camera-model-to-choose)
  * [How to obtain and verify good calibration results](#how-to-obtain-and-verify-good-calibration-results)
  * [How to use generic camera models in your application](#how-to-use-generic-camera-models-in-your-application)
  * [Reference on calibration report visualizations](#reference-on-calibration-report-visualizations)



## Overview ##

This repository contains a tool for **accurate geometric camera calibration**,
i.e., establishing a mapping between image pixels and the pixels' 3D observation
directions respectively lines. In particular, it supports calibration with
**generic** camera models, which fit nearly every camera and allow for highly
accurate calibration. The tool also includes support to calibrate
**fixed camera rigs** and additionally supports estimating
**accurate depth images for stereo cameras** such as the Intel D435 or the
Occipital Structure Core.

The requirements on the camera are:

* The camera must be near-central, i.e., all observation lines must approximately pass
  through the same 3D point. This is because a central camera model is used for initialization.
  There is support for a fully non-central model, but not for initializing directly with it
  (however, re-implementations of Ramalingam and Sturm's initialization methods for
  all combinations of central/non-central cameras and planar/non-planar calibrations patterns
  are available in [applications/camera_calibration/src/camera_calibration/relative_pose_initialization](https://github.com/puzzlepaint/camera_calibration/tree/master/applications/camera_calibration/src/camera_calibration/relative_pose_initialization)).
* The observation directions / lines must vary smoothly. For example,
  there should not be a mirror within the field-of-view of the camera that ends
  abruptly. This is because observation directions / lines are stored sparsely
  and are interpolated smoothly.

For depth estimation and live feature detection, a CUDA-capable graphics card is required.

The application has been tested on Ubuntu Linux only.



## About ##

This repository contains the
[Camera calibration application](https://github.com/puzzlepaint/camera_calibration/tree/master/applications/camera_calibration)
and the library it is based on,
[libvis](https://github.com/puzzlepaint/camera_calibration/tree/master/libvis).
The library is work-in-progress and it is not recommended to use it for other projects at this point.

The application and library code is licensed under the BSD license, but please
also notice the licenses of the included or externally used third-party components.

If you use the provided code for research, please cite the paper describing the approach:

[Thomas Schöps, Viktor Larsson, Marc Pollefeys, Torsten Sattler, "Why Having 10,000 Parameters in Your Camera Model is Better Than Twelve", arXiv 2019.](https://arxiv.org/abs/1912.02908)



## Building ##

Building has been tested on Ubuntu 14.04 and Ubuntu 18.04 (with gcc).

The following external dependencies are required.

| Dependency   | Version(s) known to work |
| ------------ | ------------------------ |
| [Boost](https://www.boost.org/) | 1.54.0 |
| [CUDA](https://developer.nvidia.com/cuda-downloads) | 10.1 |
| [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) | 3.3.7 |
| [GLEW](http://glew.sourceforge.net/build.html) | 1.10.0 |
| [OpenGV](https://github.com/laurentkneip/opengv) | Commit 306a54e6c6b94e2048f820cdf77ef5281d4b48ad |
| [Qt](https://www.qt.io/) | 5.12.0; minimum version: 5.8 |
| [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) | 4.2.1 |
| [zlib](https://zlib.net/) | - |

The following external dependencies are optional.

| Dependency   | Purpose |
| ------------ | ------- | 
| [librealsense2](https://github.com/IntelRealSense/librealsense) | Live input from RealSense D400 series depth cameras (tested with the D435 only). |
| [Structure SDK](https://structure.io/developers) | Live input from Structure Core cameras (tested with the color version only). To use this, set the SCSDK_ROOT CMake variable to the SDK path. |

After obtaining all dependencies, the application can be built with CMake, for example as follows:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_FLAGS="-arch=sm_61" ..
make -j camera_calibration  # Reduce the number of threads if running out of memory, e.g., -j3
```

If you intend to use the depth estimation or live feature detection functionalities,
make sure to specify suitable CUDA architecture(s) in CMAKE_CUDA_FLAGS.
Common settings would either be the CUDA architecture of your graphics card only (in case
you only intend to run the compiled application on the system it was compiled on), or a range of virtual
architectures (in case the compiled application is intended for distribution).
See the [corresponding CUDA documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-gpu-architecture).



## How to use ##

### Obtaining a calibration pattern ###

This is a prerequisite for calibration.

The first step is to choose a suitable pattern. Ideally, the density of features
on the pattern is chosen to be appropriate for the resolution of the camera to
be calibrated. For example, a high-resolution camera can observe many features
at the same time, so a high feature density helps in quickly obtaining enough
calibration data. However, this pattern may not be well-suited for a low-resolution
camera, which cannot sharply observe all features at the same time. It should
also be considered that high numbers of features (either due to high density, or
due to using multiple patterns at the same time) significantly increase the time
required to perform the calibration.

Some readily usable patterns with different feature densities, generated for
DIN A4 sized paper, are included in the
[patterns](https://github.com/puzzlepaint/camera_calibration/tree/master/applications/camera_calibration/patterns) folder.
Each pattern consists of a PDF file for display and a YAML file that describes the pattern content.
The YAML file later needs to be passed to the camera calibration program such that it can detect
the corresponding pattern.

If the provided patterns are not sufficient, you can
generate additional patterns with the pattern generation script [scripts/create_calibration_pattern.py](https://github.com/puzzlepaint/camera_calibration/tree/master/applications/camera_calibration/scripts/create_calibration_pattern.py). The script uses [ReportLab](https://www.reportlab.com/) to generate the PDF file,
which may be installed like: `sudo pip[3] install reportlab`. It also depends on
numpy. Call the script as follows to see its usage:
`python[3] create_calibration_pattern.py -h`. Only the `--tag36h11_path` and
`--output_base_path` arguments are mandatory.

After deciding for one or multiple patterns, the second step is to choose how to
present the pattern(s) to the camera:

* One way is to **print** the pattern(s). It is possible to use multiple printed patterns
  at the same time. This may for example help to fully calibrate fisheye cameras
  with a very large field-of-view, since the pattern geometry is not limited to
  a plane then. In this case, make sure that each printed pattern uses a unique
  AprilTag index. Note that the final pattern(s) must be rigid, fixed wrt. each
  other, and each individual pattern should be near-planar.
  Planarity is assumed for initialization purposes, but not for later stages in
  the calibration. Thus, as long as initialization works, the final accuracy is not
  negatively affected by non-planar patterns in any way.
* Another way is to **show the pattern on a display** such as a computer monitor.
  The application includes direct support to do this, while also showing the
  feature coverage while not recording images. See below for how to use this.
  This way, only a single pattern can be used at a time. If multiple patterns
  are given, the current pattern can be changed with the arrow keys.



### Calibrating a camera with live input ###

Live input has the advantage that the coverage of the camera view with feature
detections is shown in real-time during recording, showing where additional data
is still needed. However, this is only possible for cameras for which live
support has been implemented. Currently, there is support for Intel RealSense
cameras via librealsense2, for Occipital Structure Core cameras via the Structure
SDK, and for many other kinds of cameras with video4linux2.

To use this mode of operation, start the application without arguments:

```bash
/path/to/camera_calibration/build/applications/camera_calibration/camera_calibration
```

This will show a window that might look like this with a webcam and an Intel RealSense D435 camera attached:

![Settings Window](applications/camera_calibration/doc/settings_window.png?raw=true)

At the top, all attached and detected cameras are listed. They are prefixed by the
library that they are detected with. A single camera may be detected by multiple
libraries; for example, here the three cameras on the D435 device were detected
by librealsense and by video4linux2 (but in this case, they will only work with
librealsense).

In this list, check the boxes for all cameras that should be used at the same
time. Note that at present, it is only possible to check multiple "librealsense"
cameras or multiple "Structure SDK" cameras at the same time, but no other
cameras or cameras used with different libraries.

The "Live feature detection" box should remain checked to give a live image of
the image coverage with feature detections. It should be unchecked if no
CUDA-capable graphics card is available, or if recording data for other purposes.

In the text field above this box, the paths to the pattern YAML files that will
be used must be entered. If the mode which shows the pattern on screen will be used later,
this pattern must also be selected here.

The feature window extent should be set to suit the specific camera(s) used. It
is recommended to shortly try out a few different values and choose the value
which gives the most reliable feature detections. Common values are for example
10, 15, and 20.

Saving the recorded images is helpful in case you cannot run real-time feature
detection, or if you potentially want to process the images
again later with other settings. If you do not want to save the images, the
corresponding checkbox can be un-ticked.

For saving the recorded images, and a dataset file containing the features extracted in
real-time, specify a directory to save the dataset and images in at the bottom.

From here on, there are two ways to start live operation:

* Click "Show pattern" to display the selected patterns in fullscreen mode on
  the computer screen. Note that in this mode, no automatic image recording
  or live detection is done. Instead, an image is recorded and features are detected in it when pressing the
  Space key. The live camera view is displayed while no image is recorded,
  which is hidden while recording an image.
  Note that for using this mode, you must start the application with the `--apriltags_directory`
  parameter, specifying the path to a directory containing the "tag36h11" AprilTag
  images. Those can be downloaded from [the corresponding repository](https://github.com/AprilRobotics/apriltag-imgs).
* Click "Start normal" to use printed or otherwise externally shown patterns.
  This will show a window which only shows the live images and the feature
  coverage.

To end recording, simply close the recording window (use Escape or Alt+F4 in case of the
fullscreen pattern display).

Recording with live feature detection yields a file `dataset.bin` that can be further processed to calibrate
the camera as described in the second step of the section below. If only recording
images, proceed as described from the start of the section below.



### Calibrating a camera from images in a folder ###

This mode of operation may be used for cameras for which live input is not possible,
or after recording images live as described above.

#### Feature extraction ####

To extract features and create a dataset file, the camera calibration program
can be first called as follows, for example. This assumes that the images have
been placed in a folder `${DATASET}/images`.

```bash
export CALIBRATION_PATH=/path/to/camera_calibration_root_folder
export DATASET=/path/to/dataset_folder
export HALF_WINDOW_SIZE=15  # Adjust to what gives the most detections for your camera, e.g., 10, 15, or 20
${CALIBRATION_PATH}/build/applications/camera_calibration/camera_calibration \
    --pattern_files ${CALIBRATION_PATH}/applications/camera_calibration/patterns/pattern_resolution_17x24_segments_16_apriltag_0.yaml \
    --image_directories ${DATASET}/images \
    --dataset_output_path ${DATASET}/features_${HALF_WINDOW_SIZE}px.bin \
    --refinement_window_half_extent ${HALF_WINDOW_SIZE} \
    --show_visualizations  # optional for showing visualizations
#   --no_cuda_feature_detection  # use this to disable using CUDA for feature detection
```

`--pattern_files` must be a comma-separated list of paths to YAML files
describing the calibration pattern(s) used. `--image_directories` specifies the
path to the directory containing the images. If calibrating a camera rig, multiple
comma-separated folders must be given. Images in different folders that have the
same file name are assumed to be recorded at the same time. `--dataset_output_path` gives the
path to a file that will be created to store the extracted features. If you use
`--show_visualizations`, the visualization window will remain open once the process has finished and
needs to be closed manually.

#### Camera calibration ####

As a second step, the camera calibration program can be called to perform the
actual calibration based on the extracted features, for example as follows
(using the definitions from above):

```bash
export CELL_SIZE=50  # Choose a suitable value for the camera's resolution
${CALIBRATION_PATH}/build/applications/camera_calibration/camera_calibration \
    --dataset_files ${DATASET}/features_${HALF_WINDOW_SIZE}px.bin \
    --output_directory ${DATASET}/result_${HALF_WINDOW_SIZE}px_noncentral_generic_${CELL_SIZE} \
    --cell_length_in_pixels ${CELL_SIZE} \
    --model noncentral_generic \
    --num_pyramid_levels 4 \
    --show_visualizations  # optional for showing visualizations
```

`--dataset_files` must point to the dataset file with the extracted features.
The computed calibration files will be saved in the folder given with `--output_directory`.
`--cell_length_in_pixels` specifies the desired cell length for generic camera models;
see below. The camera model to use must be given with `--model`. For generic
camera models, it can be helpful to use a multi-resolution pyramid during
calibration for better convergence. The number of pyramid levels can be given
with `--num_pyramid_levels`. Note that re-sampling for the `noncentral_generic`
model is implemented in a somewhat inaccurate way, however. If you use
`--show_visualizations`, the visualization window will remain open once the
process has finished and needs to be closed manually.

The available camera models are as follows. See the corresponding section below for
recommendations on which model to choose.

```
central_generic
central_thin_prism_fisheye
central_opencv
central_radial
noncentral_generic
```

For generic camera models, a grid resolution respectively cell size must be chosen.
Calibrated 3D observation directions or lines are stored at the corners of the resulting grid
and are interpolated over the grid cells. Note that the given cell size is not used
directly; rather, the closest cell size is chosen that yields an integer number of
cells over the calibrated image area.

The grid resolution should be chosen to be appropriate for the camera's resolution.
For example, for a camera of resolution 2000x1000 pixels, a cell length of 40 might
be appropriate, while for a camera of resolution 640x480 pixels, a cell length of 10
might be appropriate. The points to consider are:

* Denser grids may better model small details, improving the calibration.
* To properly constrain the camera model parameters, multiple feature observations
  should be recorded within each grid cell. I.e., with a denser grid, denser
  feature observations are required to properly constrain the model and avoid
  overfitting.
* Denser grids increase the time required to calibrate the model. However, the
  runtime performance impact when using the calibrated model should be negligible.

The output files contain some "report" files that allow to judge the quality of
the resulting calibration. See the section "How to obtain good calibration results"
below.

#### Refining existing calibrations ####

It is also possible to take an existing calibration and refine it, possibly after
re-sampling to a different camera model. To do this, run the calibration program
as specified above, but also give the directory in which the existing calibration
is saved in with the `--state_directory` parameter. Note that re-sampling camera
models is only implemented between different central models, from a central model
to the non-central model, and (approximately) from the non-central model to a
different grid resolution, but not from the non-central model to a central model.
For example, for near-central cameras, this allows to calibrate the camera with
a central model first and then use the non-central model as last refinement step.

#### Handling large datasets and many variables ####

The application computes the Schur complement during bundle adjustment while
solving for state updates. By default, it will fully store the off-diagonal
part of the Hessian matrix in memory for its computation, which may become huge
if there are many images and thus many pose variables to be optimized, as well
as many intrinsics variables to be optimized. This may be very slow and/or
exceed the available memory. To better handle such cases, the program allows to
change this behavior by specifying the `--schur_mode` parameter. It supports the
following options:

* **dense**: This is the default behavior. Use this if you have sufficient
  memory and cannot use the CUDA variant (or the latter does not help).
* **dense_cuda**: Performs a large matrix multiplication during computation of
  the Schur complement on the GPU with CUDA. This may improve the performance,
  but does not reduce memory use.
* **dense_onthefly**: Stores only a few rows of the off-diagonal part of the
  Hessian at each time. The rows are stored densely. This requires more passes
  over the residuals than the default option and is slower than it, but saves
  memory.
* **sparse**: Stores the off-diagonal part of the Hessian sparsely (but keeps
  it in memory completely). This may be faster than the default if the matrix
  is very sparse, and may potentially save some memory.
* **sparse_onthefly**: Stores only a few rows of the off-diagonal part of the
  Hessian at each time. The rows are stored sparsely. This requires more passes
  over the residuals than the default option, but saves memory. It might be
  well-suited if the matrix is very sparse.

You may need to try out which option works best for your case. If you do not
run into any issues with memory or performance, you may simply leave this
option at its default.



### Calibrating a stereo camera and computing depth images ###

This requires a fixed configuration of two cameras whose fields of view overlap.
For example, this is well-suited to calibrate active stereo cameras such as the
Intel D435 or the Occipital Structure Core. However, it is also possible to
put two arbitrary individual cameras next to each other to make a stereo rig.
Note that this configuration needs to remain completely fixed though for the calibration
to remain valid, and both cameras are supposed to take images at exactly the same
time; alternatively, the scene must be static, such that different recording times do
not matter.

Also note that at the moment, this supports only a single camera model at a time,
depending on which model the CUDA kernel for stereo depth estimation is compiled with.
See [libvis/src/libvis/cuda/pixel_corner_projector.cuh](https://github.com/puzzlepaint/camera_calibration/tree/master/libvis/src/libvis/cuda/pixel_corner_projector.cuh).
By default, it is the central-generic camera model.

Another limitation of the implementation (that should be trivial to fix if required)
is that the calibration must have been made with exactly the two cameras that
will be used for stereo depth estimation (and no additional ones).

If using an active stereo camera, the active projection should be disabled for
calibration. The librealsense integration can do this if using a RealSense
camera for live input. For other cameras, the projector needs to be covered to
block the light.

Calibration otherwise works as described in the sections above, either with live
camera input or based on recorded images.

For depth estimation, stereo images with the active projection turned on should
be recorded. Depth maps can then be computed for example as follows:

```bash
export CALIBRATION_PATH=/path/to/camera_calibration
export CALIBRATION_RESULT=/path/to/calibration/result/folder
export STEREO_DATASET=/path/to/input/image/dataset
export IMAGE=image_filename_without_png
${CALIBRATION_PATH}/build/applications/camera_calibration/camera_calibration \
    --stereo_depth_estimation \
    --state_directory ${CALIBRATION_RESULT} \
    --images ${STEREO_DATASET}/images0/${IMAGE}.png,${STEREO_DATASET}/images1/${IMAGE}.png \
    --output_directory ${STEREO_DATASET}/stereo_${IMAGE}
```

This assumes that the stereo images have been recorded with the camera_calibration
program, which places the images of the two cameras in the `images0` and `images1` folders.

Note that the stereo depth estimation implementation has not at all been optimized
and may thus take a very long time to compute.



### Which camera model to choose? ###

For best results, choose one of the following models:

* `central_generic` for assuming a central camera (all observation rays go through a single point), or
* `noncentral_generic` for general non-central cameras.

Usually, `noncentral_generic` is slightly more accurate than `central_generic`,
even for near-central cameras. In general, it should always be at least as
accurate as `central_generic`, unless a lack of data leads to overfitting.

However, one should be aware of the implications: With a non-central camera model,
images in general cannot be undistorted to pinhole images (without knowing the
scene geometry), and algorithms developed for central cameras might require adaptation.
For this reason, using a central camera model might be more convenient, even if
being a little less accurate.



### How to obtain and verify good calibration results ###

Some tips to follow for getting good calibration results are:

* Show the calibration pattern to the camera(s) from all angles and from different distances.
  The calibration will fail if the pattern is only visible from straight above.
  Showing the effect of perspective is necessary to calibrate the camera field-of-view.
* Cover the whole camera image with feature detections. In particular, focus on the image corners,
  as it is hardest to get sufficient detections there. If using a generic camera model,
  each grid cell should contain several feature detections.
* If using a rolling shutter camera, use a tripod and only take images from fixed
  poses (not hand-held) to avoid introducing any rolling shutter distortions.

After computing a calibration, the report files within the output directory
allow judging the calibration quality.

* In `report_cameraX_info.txt`, `reprojection_error_median` should usually be
  significantly smaller than 0.1 pixels.
* `report_cameraX_errors_histogram.png` should be a (more or less) small white
  and round(-ish) dot in the center of the image, such as:
  
  ![Error histogram good example](applications/camera_calibration/doc/report_cameraX_errors_histogram_good_example_1.png?raw=true)
  
  This is another good example with a larger dot:
  
  ![Error histogram good example](applications/camera_calibration/doc/report_cameraX_errors_histogram_good_example_2.png?raw=true)
  
  If the dot is not round or is not centered, something is definitely wrong. Example:
  
  ![Error histogram good example](applications/camera_calibration/doc/report_cameraX_errors_histogram_bad_example_1.png?raw=true)
  
  Another bad example:
  
  ![Error histogram good example](applications/camera_calibration/doc/report_cameraX_errors_histogram_bad_example_2.png?raw=true)
  
  A possible reason for such failure cases is that the bundle adjustment is not converged yet and needs more iterations.
  It could also be that the selected camera model does not fit to the camera at all, but that should be unlikely to happen if using a suitable generic model.
* Most cells in `report_cameraX_error_magnitudes.png` should be more or less
  green (if using outlier removal). If there are red points forming some systematic
  pattern, something is probably wrong.
* `report_camera0_error_directions.png` should show random colors. If there is
  a systematic pattern, then the calibrated model does not fit the data tightly
  (this is bound to happen if using parametric camera models!).
  Good example (generic camera model):
  
  ![Error histogram good example](applications/camera_calibration/doc/report_cameraX_error_directions_good_example.jpg?raw=true)
  
  Bad example (parametric camera model):
  
  ![Error histogram good example](applications/camera_calibration/doc/report_cameraX_error_directions_bad_example_1.jpg?raw=true)
  
  Note that all kinds of different systematic patterns can show up here. Also,
  even in good calibrations, weak patterns may remain. Another failure mode is
  missing projections in areas where features were detected. This shows up as
  areas of large Voronoi cells; example in the top-left corner:
  
  ![Error histogram good example](applications/camera_calibration/doc/report_cameraX_error_directions_bad_example_2.jpg?raw=true)
  
  This kind of failure probably means that the optimization process
  used for projection does not find the optimum for points that project to this
  area. This may be caused by unusual camera geometry, or by having too much noise
  in the calibration, possibly caused by not enough feature detections.
  For the example above, we can confirm the case of a noisy calibration by
  zooming in on the top-left corner of the observation direction visualization:
  
  ![Error histogram good example](applications/camera_calibration/doc/noisy_calibration_example.png?raw=true)
  
  Here, the faint blue rims change their direction at the top left corner (zoom into the image to see this better).
  This creates a "trap" for the optimization from which it cannot escape,
  causing points near this corner to fail projection (since the optimization
  would first overshoot and then go back, but it fails to go back in this case).



### How to use generic camera models in your application ###

After successful calibration, the calibrated intrinsic camera parameters are
stored in the files `intrinsicsX.yaml` in the output folder.

In the [applications/camera_calibration/generic_models](https://github.com/puzzlepaint/camera_calibration/tree/master/applications/camera_calibration/generic_models) folder,
there are implementations for the central-generic and non-central generic camera
models which can load these intrinsics YAML files. This should make it easy
to use these camera models in other applications. These implementations support
3D point projection to the image, pixel un-projection to a 3D direction respectively
line, and computing Jacobians for the above operations with respect to the input
point or pixel.

These camera model implementations use the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) library as a single dependency.
Even this dependency should be easy to remove if desired, since only its
matrix and vector classes are used, but no advanced functionality that would be
hard to substitute. See the [main file](https://github.com/puzzlepaint/camera_calibration/tree/master/applications/camera_calibration/generic_models/src/main.cc)
of this implementation for some unit tests, which show by example how to use the
camera model classes. The camera models are also documented with Doxygen comments.
However, note that these implementations have not been optimized; depending on the
application, it could be sensible to use different kinds of lookup tables to
speed up the operations.

Note that the calibration program will not calibrate the whole image area, but
only the bounding rectangle of all feature detections. Due to the local window
size for feature refinement, features are not detected directly next to the image
borders. If it was crucial to calibrate the whole image area, it would for example
be possible to extrapolate the calibration, or to tolerate some overlap of the feature
refinement window with regions outside of the image.



### Reference on calibration report visualizations ###

* `report_cameraX_error_directions.png`: Each pixel in this image is colored
  according to the *direction* (disregarding the magnitude) of the reprojection
  error of the closest residual (over all images used for calibration). This
  allows judging whether there are any systematic patterns in the residual
  error directions, even very small ones. This visualization is a Voronoi
  diagram. It also allows judging whether there are too few feature detections
  in some part of the image; those cause large Voronoi cells.
* `report_cameraX_error_magnitudes.png`: Each pixel in this image is colored
  according to the *magnitude* (disregarding the direction) of the reprojection
  error of the closest residual (over all images used for calibration). Low
  errors are colored green, high errors are colored red.
* `report_cameraX_errors_histogram.png`: Shows a 2D histogram of all reprojection
  errors. Allows judging whether the residual distribution is as expected (dot-shaped).
* `report_cameraX_grid_point_locations.png`: Shows the locations of the grid points for
  generic camera models that use a grid for interpolation.
* `report_cameraX_line_offsets.png`: For the non-central-generic model, this image
  visualizes the positions of the observation lines as follows: First, a 3D point
  is determined which is as close as possible to all observation lines. For a
  central camera, this would be the projection center. Then, for each pixel, the
  closest point on the pixel's observation line to this 3D point is determined.
  The 3D offset between those two points is directly translated into an RGB color
  for this pixel in the visualization. This visualization allows judging whether
  the lines follow some clear pattern (which suggests that the camera is
  significantly non-central), or appear more or less random (which suggests that
  the camera is mostly central). Note that the automatic scaling will usually
  cause almost all areas of this visualization to be gray (since the extrema
  will usually be only in small parts of the image). The contrast can be changed
  with an image editing program such as GIMP to see the remaining structure.
* `report_cameraX_observation_directions.png`: Visualizes the calibrated observation directions.
  Each 3D direction is directly mapped to an RGB color in the visualization.
  More structure is shown for the z direction, since by convention, this is
  calibrated to be the 'forward' direction for each camera, and too little
  structure might be visible if treating it the same as the other two dimensions.
