// Copyright 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <Eigen/Geometry>
#include <libvis/camera.h>
#include <libvis/command_line_parser.h>
#include <libvis/eigen.h>
#include <libvis/external_io/colmap_model.h>
#include <libvis/geometry.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/point_cloud.h>
#include <QSharedPointer>
#include <QtWidgets>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/bundle_adjustment/joint_optimization.h"
#include "camera_calibration/calibration.h"
#include "camera_calibration/calibration_report.h"
#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"
#include "camera_calibration/fitting_report.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/image_input/image_input_realsense.h"
#include "camera_calibration/image_input/image_input_v4l2.h"
#include "camera_calibration/models/all_models.h"
#include "camera_calibration/tools/tools.h"
#include "camera_calibration/ui/calibration_window.h"
#include "camera_calibration/ui/live_image_consumer.h"
#include "camera_calibration/ui/main_window.h"
#include "camera_calibration/ui/pattern_display.h"
#include "camera_calibration/ui/settings_window.h"
#include "camera_calibration/util.h"

using namespace vis;


Q_DECLARE_METATYPE(QSharedPointer<Image<Vec3u8>>)

int LIBVIS_QT_MAIN(int argc, char** argv) {
  qRegisterMetaType<QSharedPointer<Image<Vec3u8>>>();
  srand(0);
  
  
  // Parse command line arguments
  CommandLineParser cmd_parser(argc, argv);
  
  
  // ### Feature extraction ###
  vector<string> image_directories;
  cmd_parser.NamedPathParameter(
      "--image_directories", &image_directories, ',', /*required*/ false,
      "Comma-separated list of paths to folders containing pre-recorded images to use for calibration. If multiple folders are given, images with the same filename are assumed to have been recorded simultaneously by a fixed camera rig.");
  
  vector<string> pattern_files;
  cmd_parser.NamedPathParameter(
      "--pattern_files", &pattern_files, ',', /*required*/ false,
      "Comma-separated list of paths to YAML files describing the calibration patterns used.");
  
  string dataset_output_path = "";
  cmd_parser.NamedPathParameter(
      "--dataset_output_path", &dataset_output_path, /*required*/ false,
      "Path to a YAML or bin file where the extracted features for all images will be stored.");
  
  int refinement_window_half_extent = 10;
  cmd_parser.NamedParameter(
      "--refinement_window_half_extent", &refinement_window_half_extent, /*required*/ false,
      "Half size of the search window for pattern corner features. There is a tradeoff here: on the one hand, it should be as small as possible to be able to detect features close to the image borders and include little distortion. On the other hand, it needs to be large enough to be able to detect features properly. Especially if corners are blurred, it is necessary to increase this extent.");
  
  string feature_refinement_type = "intensities";
  cmd_parser.NamedParameter(
      "--feature_refinement_type", &feature_refinement_type, /*required*/ false,
      "The type of feature refinement. Possible values: gradients_xy, gradient_magnitude, intensities, no_refinement");
  
  bool cuda_feature_detection = !cmd_parser.Flag(
      "--no_cuda_feature_detection", "Disable CUDA in feature detection. CUDA feature detection only supports the gradients_xy and intensities refinement types, and might be slightly less accurate than CPU feature detection.");
  
  
  // ### Calibration ###
  // Inputs / cache
  vector<string> dataset_files;
  cmd_parser.NamedPathParameter(
      "--dataset_files", &dataset_files, ',', /*required*/ false,
      "Comma-separated list of paths to YAML or bin files (saved using --dataset_output_path or created externally) with pre-extracted features to use for calibration.");
  
  string state_directory = "";
  cmd_parser.NamedPathParameter(
      "--state_directory", &state_directory, /*required*/ false,
      "Path to a folder containing files intrinsicsX.yaml, pattern.yaml, camera_tr_rig.yaml and rig_tr_global.yaml, used to initialize the optimization state.");
  
  string dense_initialization_base_path = "";
  cmd_parser.NamedPathParameter(
      "--dense_initialization_base_path", &dense_initialization_base_path, /*required*/ false,
      "Base path of a pair of files that will be written (if they do not exist yet) or read (if they exist) to cache the calibration initialization. Useful to quickly test different settings for the later calibration stages without having to repeat the initialization each time.");
  
  // Outputs
  string output_directory = "";
  cmd_parser.NamedPathParameter(
      "--output_directory", &output_directory, /*required*/ false,
      "Directory to store calibration outputs in, with default naming. Sets --state_output_directory, --pruned_dataset_output_path, and --report_base_path.");
  
  string state_output_directory = output_directory;
  cmd_parser.NamedPathParameter(
      "--state_output_directory", &state_output_directory, /*required*/ false,
      "Path to the directory where the computed calibration will be saved.");
  
  string pruned_dataset_output_path = output_directory.empty() ? "" : (boost::filesystem::path(output_directory) / "dataset.bin").string();
  cmd_parser.NamedPathParameter(
      "--pruned_dataset_output_path", &pruned_dataset_output_path, /*required*/ false,
      "Path to a YAML or bin file where the dataset after calibration will be stored (i.e., potentially having outliers removed).");
  
  string report_base_path = output_directory.empty() ? "" : (boost::filesystem::path(output_directory) / "report").string();
  cmd_parser.NamedPathParameter(
      "--report_base_path", &report_base_path, /*required*/ false,
      "Base filename for storing the calibration error report files.");
  
  // Settings
  string model_string = "";
  cmd_parser.NamedParameter(
      "--model", &model_string, /*required*/ false,
      "Name of the camera model to use. One of: central_generic, central_thin_prism_fisheye, central_opencv, central_radial, noncentral_generic.");
  
  float outlier_removal_factor = 6;
  cmd_parser.NamedParameter(
      "--outlier_removal_factor", &outlier_removal_factor, /*required*/ false,
      "Specifies the factor in the term defining the outlier removal threshold: third_quartile_error + factor * (third_quartile_error - first_quartile_error). Set to a value less or equal to zero to disable outlier removal.");
  
  int cell_length_in_pixels = 25;
  cmd_parser.NamedParameter(
      "--cell_length_in_pixels", &cell_length_in_pixels, /*required*/ false,
      "Approximate (!) cell side length in pixels. Will be slightly adjusted such that the calibrated image area size is an integer multiple of the cell size.");
  
  float regularization_weight = 0;
  cmd_parser.NamedParameter(
      "--regularization_weight", &regularization_weight, /*required*/ false,
      "Weight of the regularization terms in the cost function. TODO: Regularization is currently disabled, so this setting has no effect.");
  
  int num_pyramid_levels = 3;
  cmd_parser.NamedParameter(
      "--num_pyramid_levels", &num_pyramid_levels, /*required*/ false,
      "Number of multi-resolution pyramid levels to use for the resolution of the intrinsics grid in bundle adjustment. More levels may improve convergence. Less levels make it applicable to less smooth cameras. Setting this to 1 uses only the full resolution. Different settings from 1 are only sensible for (generic) camera models with a grid.");
  
  bool localize_only = cmd_parser.Flag(
      "--localize_only", "Load an existing intrinsics calibration (from --state_directory), keep it fixed, and optimize for camera poses and pattern geometry only.");
  
  bool show_visualizations = cmd_parser.Flag(
      "--show_visualizations", "Show some visualizations during calibration");
  
  string apriltags_directory = "";
  cmd_parser.NamedParameter(
      "--apriltags_directory", &apriltags_directory, /*required*/ false,
      "Path to a directory containing the \"tag36h11\" AprilTag images. This is required for showing the pattern on the monitor for data recording.");
  
  string schur_mode_string = "";
  cmd_parser.NamedParameter(
      "--schur_mode", &schur_mode_string, /*required*/ false,
      "Specifies how to compute the Schur complement for matrix solving in bundle adjustment, affecting performance and memory use. Must be one of: dense, dense_cuda, dense_onthefly, sparse, sparse_onthefly.");
  
  
  // ### Visualizations ###
  bool create_calibration_report = cmd_parser.Flag(
      "--create_calibration_report", "Run the tool to create a calibration report for a calibrated state");
  
  bool create_fitting_visualization = cmd_parser.Flag(
      "--create_fitting_visualization", "Run the tool to create a model fitting visualization");
  
  float max_visualization_extent = -1;
  cmd_parser.NamedParameter(
      "--max_visualization_extent", &max_visualization_extent, /*required*/ false,
      "Setting for --create_fitting_visualization; -1 for automatic choice.");
  
  float max_visualization_extent_pixels = -1;
  cmd_parser.NamedParameter(
      "--max_visualization_extent_pixels", &max_visualization_extent_pixels, /*required*/ false,
      "Setting for --create_fitting_visualization; -1 for automatic choice.");
  
  
  // ### File conversion ###
  bool convert_dataset = cmd_parser.Flag(
      "--convert_dataset", "Run the tool to convert a dataset file");
  
  
  // ### Calibration comparison ###
  bool compare_calibrations = cmd_parser.Flag(
      "--compare_calibrations", "Run the tool to compare two existing calibrations");
  
  string calibration_a = "";
  cmd_parser.NamedPathParameter(
      "--calibration_a", &calibration_a, /*required*/ false,
      "Setting for --compare_calibrations: path to calibration.yaml of the first compared calibration.");
  
  string calibration_b = "";
  cmd_parser.NamedPathParameter(
      "--calibration_b", &calibration_b, /*required*/ false,
      "Setting for --compare_calibrations: path to calibration.yaml of the second compared calibration.");
  
  
  // ### Kalibr calibration visualization ###
  bool visualize_kalibr_calibration = cmd_parser.Flag(
      "--visualize_kalibr_calibration", "Run the tool to visualize a calibration made by Kalibr.");
  
  string camchain_path = "";
  cmd_parser.NamedPathParameter(
      "--camchain_path", &camchain_path, /*required*/ false,
      "Path to the Kalibr camchain for --visualize_kalibr_calibration.");
  
  
  // ### Colmap calibration visualization ###
  bool visualize_colmap_calibration = cmd_parser.Flag(
      "--visualize_colmap_calibration", "Run the tool to visualize a calibration made by Colmap.");
  
  string cameras_path = "";
  cmd_parser.NamedPathParameter(
      "--cameras_path", &cameras_path, /*required*/ false,
      "Path to the Colmap cameras.txt for --visualize_colmap_calibration.");
  
  
  // ### Synthetic dataset rendering ###
  bool render_synthetic_dataset = cmd_parser.Flag(
      "--render_synthetic_dataset", "Run the tool to render a synthetic dataset.");
  
  string synthetic_dataset_files = "";
  cmd_parser.NamedPathParameter(
      "--synthetic_dataset_files", &synthetic_dataset_files, /*required*/ false,
      "Path to the directory of the dataset created by --render_synthetic_dataset.");
  
  
  // ### Dataset intersection ###
  bool intersect_datasets = cmd_parser.Flag(
      "--intersect_datasets", "Run the tool to intersect a number of datasets, i.e., keep only features that are detected in all of them. The assumption is that all datasets contain images taken from the same poses, with a possibly different pattern / feature detector.");
  
  float intersection_threshold = 3.0;
  cmd_parser.NamedParameter(
      "--intersection_threshold", &intersection_threshold, /*required*/ false,
      "Maximum distance in pixels between two feature detections to be considered equal for --intersect_datasets.");
  
  
  // ### Stereo depth estimation ###
  bool stereo_depth_estimation = cmd_parser.Flag(
      "--stereo_depth_estimation", "Run dense stereo depth estimation on some images of a camera rig with a given calibration.");
  
  vector<string> images;
  cmd_parser.NamedParameter(
      "--images", &images, ',', /*required*/ false,
      "Comma-separated list of paths to images for --stereo_depth_estimation. Must be ordered according to the ordering of cameras in the used state.");
  
  
  // ### Point cloud comparison ###
  bool compare_point_clouds = cmd_parser.Flag(
      "--compare_point_clouds", "Run the tool to compare point clouds created with dense stereo and different camera models / settings.");
  
  string stereo_directory_target = "";
  cmd_parser.NamedParameter(
      "--stereo_directory_target", &stereo_directory_target, /*required*/ false,
      "First directory with stereo results for --compare_point_clouds. This will be the target for point cloud matching with a similarity transform.");
  
  string stereo_directory_source = "";
  cmd_parser.NamedParameter(
      "--stereo_directory_source", &stereo_directory_source, /*required*/ false,
      "Second directory with stereo results for --compare_point_clouds.");
  
  // ### Legend creation ###
  bool create_legends = cmd_parser.Flag(
      "--create_legends", "Run the tool to create legends for the visualizations.");
  
  // ### Localization accuracy test ###
  bool localization_accuracy_test = cmd_parser.Flag(
      "--localization_accuracy_test", "Run the localization accuracy test tool");
  
  string localization_accuracy_gt_model = "";
  cmd_parser.NamedPathParameter(
      "--localization_accuracy_gt_model", &localization_accuracy_gt_model, /*required*/ false,
      "Path to the YAML file of the ground truth camera model for --localization_accuracy_test.");
  
  string localization_accuracy_compared_model = "";
  cmd_parser.NamedPathParameter(
      "--localization_accuracy_compared_model", &localization_accuracy_compared_model, /*required*/ false,
      "Path to the YAML file of the compared camera model for --localization_accuracy_test.");
  
  // ### Bundle adjustment ###
  bool bundle_adjustment = cmd_parser.Flag(
      "--bundle_adjustment", "Run bundle adjustment on a sparse model (stored in text format) that was initially reconstructed by the COLMAP program.");
  
  string colmap_model_path = "";
  cmd_parser.NamedPathParameter(
      "--colmap_model_path", &colmap_model_path, /*required*/ false,
      "Path to the directory containing the textual COLMAP model for use with --bundle_adjustment.");
  
  // ### Reconstruction comparison ###
  bool compare_reconstructions = cmd_parser.Flag(
      "--compare_reconstructions", "Run the tool to compare reconstructions made with --bundle_adjustment.");
  
  string reconstruction_path_1 = "";
  cmd_parser.NamedPathParameter(
      "--reconstruction_path_1", &reconstruction_path_1, /*required*/ false,
      "Path to the directory containing the first reconstruction for --compare_reconstructions.");
  
  string reconstruction_path_2 = "";
  cmd_parser.NamedPathParameter(
      "--reconstruction_path_2", &reconstruction_path_2, /*required*/ false,
      "Path to the directory containing the second reconstruction for --compare_reconstructions.");
  
  
  if (!cmd_parser.CheckParameters()) {
    return EXIT_FAILURE;
  }
  
  if (localize_only && state_directory.empty()) {
    LOG(ERROR) << "If --localize_only is used, --state_directory must be specified to load existing camera intrinsics.";
    return EXIT_FAILURE;
  }
  
  FeatureRefinement refinement_type;
  if (feature_refinement_type == string("gradients_xy")) {
    refinement_type = FeatureRefinement::GradientsXY;
  } else if (feature_refinement_type == string("gradient_magnitude")) {
    refinement_type = FeatureRefinement::GradientMagnitude;
  } else if (feature_refinement_type == string("intensities")) {
    refinement_type = FeatureRefinement::Intensities;
  } else if (feature_refinement_type == string("no_refinement")) {
    refinement_type = FeatureRefinement::NoRefinement;
  } else {
    LOG(ERROR) << "Unsupported feature refinement type: " << feature_refinement_type;
    return EXIT_FAILURE;
  }
  
  SchurMode schur_mode;
  if (schur_mode_string.empty() ||
      schur_mode_string == "dense") {
    schur_mode = SchurMode::Dense;
  } else if (schur_mode_string == "dense_cuda") {
    schur_mode = SchurMode::DenseCUDA;
  } else if (schur_mode_string == "dense_onthefly") {
    schur_mode = SchurMode::DenseOnTheFly;
  } else if (schur_mode_string == "sparse") {
    schur_mode = SchurMode::Sparse;
  } else if (schur_mode_string == "sparse_onthefly") {
    schur_mode = SchurMode::SparseOnTheFly;
  } else {
    LOG(ERROR) << "Unsupported Schur computation mode: " << schur_mode_string;
    return EXIT_FAILURE;
  }
  
  
  // Invoke a tool?
  if (compare_calibrations) {
    return CompareCalibrations(calibration_a, calibration_b, report_base_path);
  } else if (convert_dataset) {
    CHECK_EQ(dataset_files.size(), 1);
    return ConvertDataset(dataset_files[0], state_output_directory);
  } else if (create_calibration_report) {
    return WrapQtEventLoopAround([&](int /*argc*/, char** /*argv*/) {
      CHECK_EQ(dataset_files.size(), 1);
      Dataset dataset(1);
      if (!LoadDataset(dataset_files[0].c_str(), &dataset)) {
        return EXIT_FAILURE;
      }
      BAState state;
      if (!LoadBAState(state_directory.c_str(), &state, &dataset)) {
        return EXIT_FAILURE;
      }
      return CreateCalibrationReport(
          dataset,
          state,
          report_base_path);
    }, argc, argv);
  } else if (create_fitting_visualization) {
    BAState state;
    if (!LoadBAState(state_directory.c_str(), &state, nullptr)) {
      return EXIT_FAILURE;
    }
    return CreateFittingVisualization(
        state,
        report_base_path,
        max_visualization_extent,
        max_visualization_extent_pixels);
  } else if (visualize_kalibr_calibration) {
    return VisualizeKalibrCalibration(camchain_path);
  } else if (visualize_colmap_calibration) {
    return VisualizeColmapCalibration(cameras_path);
  } else if (render_synthetic_dataset) {
    return RenderSyntheticDataset(argv[0], synthetic_dataset_files);
  } else if (intersect_datasets) {
    return WrapQtEventLoopAround([&](int /*argc*/, char** /*argv*/) {return IntersectDatasets(dataset_files, intersection_threshold);}, argc, argv);
  } else if (stereo_depth_estimation) {
    return WrapQtEventLoopAround([&](int /*argc*/, char** /*argv*/) {return StereoDepthEstimation(state_directory, images, output_directory);}, argc, argv);
  } else if (compare_point_clouds) {
    return WrapQtEventLoopAround([&](int /*argc*/, char** /*argv*/) {return ComparePointClouds(stereo_directory_target, stereo_directory_source, output_directory);}, argc, argv);
  } else if (create_legends) {
    return CreateLegends();
  } else if (localization_accuracy_test) {
    return LocalizationAccuracyTest(localization_accuracy_gt_model.c_str(), localization_accuracy_compared_model.c_str());
  } else if (bundle_adjustment) {
    return BundleAdjustment(state_directory, colmap_model_path, output_directory);
  } else if (compare_reconstructions) {
    return CompareReconstructions(reconstruction_path_1, reconstruction_path_2);
  }
  
  // Prepare for batch or live calibration / feature extraction
  bool feature_extraction_requested = !image_directories.empty();
  bool calibration_requested = !localize_only && (!state_output_directory.empty() || !pruned_dataset_output_path.empty() || !report_base_path.empty());
  
  if (feature_extraction_requested && pattern_files.empty()) {
    LOG(ERROR) << "The pattern YAML file must be given with --pattern_files for feature extraction.";
    return EXIT_FAILURE;
  }
  
  CameraModel::Type model_type = CameraModel::Type::InvalidType;
  if (calibration_requested) {
    if (model_string == string("central_generic") || model_string == string("central_generic_bspline")) {
      model_type = CameraModel::Type::CentralGeneric;
    } else if (model_string == string("central_thin_prism_fisheye")) {
      model_type = CameraModel::Type::CentralThinPrismFisheye;
    } else if (model_string == string("central_opencv")) {
      model_type = CameraModel::Type::CentralOpenCV;
    } else if (model_string == string("central_radial")) {
      model_type = CameraModel::Type::CentralRadial;
    } else if (model_string == string("noncentral_generic") || model_string == string("noncentral_generic_bspline")) {
      model_type = CameraModel::Type::NoncentralGeneric;
    } else if (model_string.empty()) {
      LOG(ERROR) << "The camera model to use must be given with --model.";
      return EXIT_FAILURE;
    } else {
      LOG(ERROR) << "Camera model (given with --model) not recognized: " << model_string;
      return EXIT_FAILURE;
    }
  }
  
  shared_ptr<FeatureDetectorTaggedPattern> detector;
  if (!image_directories.empty() ||
      (image_directories.empty() && dataset_files.empty())) {
    detector.reset(new FeatureDetectorTaggedPattern(pattern_files, refinement_window_half_extent, refinement_type, cuda_feature_detection));
  }
  
  if (!image_directories.empty() || !dataset_files.empty()) {
    // Run extraction / calibration in automatic mode
    return BatchCalibrationWithGUI(
        argc,
        argv,
        image_directories,
        dataset_files,
        dense_initialization_base_path,
        state_directory,
        dataset_output_path,
        state_output_directory,
        pruned_dataset_output_path,
        report_base_path,
        detector.get(),
        num_pyramid_levels,
        model_type,
        cell_length_in_pixels,
        regularization_weight,
        outlier_removal_factor,
        localize_only,
        schur_mode,
        show_visualizations);
  } else {
    // No command-line tool was invoked, thus show the recording GUI.
    QApplication qapp(argc, argv);
    QCoreApplication::setOrganizationName("ETH");
    QCoreApplication::setOrganizationDomain("eth3d.net");
    QCoreApplication::setApplicationName("Camera Calibration");
    
    // Show the settings window at the start.
    SettingsWindow settings_window(detector, nullptr);
    settings_window.exec();
    if (settings_window.result() == QDialog::Rejected) {
      return EXIT_SUCCESS;
    }
    
    string pattern_yaml_paths = settings_window.PatternYAMLPaths();
    vector<string> pattern_yaml_paths_vector;
    if (!pattern_yaml_paths.empty()) {
      for (const QString& path : QString::fromStdString(pattern_yaml_paths).split(',')) {
        pattern_yaml_paths_vector.push_back(path.toStdString());
      }
    }
    detector->SetPatternYAMLPaths(pattern_yaml_paths_vector);
    
    refinement_window_half_extent = settings_window.FeatureWindowExtent();
    detector->SetFeatureWindowHalfExtent(refinement_window_half_extent);
    
    vector<shared_ptr<AvailableInput>> inputs = settings_window.GetChosenInputs();
    
    // Create the main window.
    MainWindow* main_window = nullptr;
    PatternDisplay* pattern_display = nullptr;
    if (settings_window.show_pattern_clicked()) {
      pattern_display = new PatternDisplay(inputs, detector, apriltags_directory.c_str(), /*TODO*/ true);
      pattern_display->showFullScreen();
      pattern_display->raise();
    } else {
      main_window = new MainWindow(inputs, nullptr, Qt::WindowFlags());
      main_window->showMaximized();
      main_window->raise();
    }
    
    // Create the (empty) dataset.
    Dataset dataset(/*num_cameras*/ inputs.size());
    std::mutex dataset_mutex;
    
    // Load the known geometries of the calibration patterns that are used.
    dataset.ExtractKnownGeometries(*detector);
    
    // If recording images, create the image directories.
    vector<QDir> image_record_directories(inputs.size());
    for (usize i = 0; i < inputs.size(); ++ i) {
      ostringstream dirname;
      if (inputs.size() == 1) {
        dirname << "images";
      } else {
        dirname << "images" << i;
      }
      image_record_directories[i] = QDir(QDir(settings_window.RecordDirectory()).absoluteFilePath(dirname.str().c_str()));
    }
    
    // Write information about the image input into dataset.yaml.
    if (settings_window.RecordImages() || settings_window.SaveDatasetOnExit() || settings_window.show_pattern_clicked()) {
      QDir dataset_dir = QDir(settings_window.RecordDirectory());
      dataset_dir.mkpath(".");
      std::string dataset_yaml_path = dataset_dir.absoluteFilePath("dataset.yaml").toStdString();
      std::ofstream dataset_yaml_file(dataset_yaml_path.c_str(), std::ios::out);
      if (dataset_yaml_file) {
        for (usize i = 0; i < inputs.size(); ++ i) {
          dataset_yaml_file << "- camera: \"" << inputs[i]->display_text.toStdString() << "\"" << endl;
          dataset_yaml_file << "  path: \"" << image_record_directories[i].dirName().toStdString() << "\"" << endl;
        }
      } else {
        LOG(ERROR) << "Failed to write dataset YAML file at: " << dataset_yaml_path;
      }
    }
    
    // Create the handler ("consumer") for new incoming images.
    LiveImageConsumer forwarder(
        settings_window.LiveDetection(),
        detector.get(),
        &dataset,
        &dataset_mutex,
        main_window ? main_window : pattern_display->main_window(),
        settings_window.RecordImages(),
        settings_window.RecordImagesWithDetectionsOnly(),
        image_record_directories);
    if (pattern_display) {
      pattern_display->SetLiveImageConsumer(&forwarder);
    }
    
    // Check whether the chosen inputs are compatible with each other. If yes,
    // create a suitable ImageInput, which starts image aquisition in the background.
    shared_ptr<ImageInput> image_input(ImageInput::CreateForInputs(&forwarder, &inputs, &settings_window));
    if (!image_input) {
      return EXIT_FAILURE;
    }
    
    // Run the application.
    qapp.exec();
    
    // Make sure that all threads exit (and no changes are being done to the
    // dataset anymore) before saving the dataset file.
    image_input.reset();
    
    if (settings_window.SaveDatasetOnExit() || settings_window.show_pattern_clicked()) {
      QDir record_directory = QDir(settings_window.RecordDirectory());
      record_directory.mkpath(".");
      SaveDataset(record_directory.absoluteFilePath("features.bin").toStdString().c_str(), dataset);
    }
    
    delete main_window;
    delete pattern_display;
    return EXIT_SUCCESS;
  }
}
