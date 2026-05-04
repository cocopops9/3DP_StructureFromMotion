#include "features_matcher.h"

#include <iostream>
#include <map>
#include <fstream>
#include <stdexcept>
#include <boost/filesystem.hpp>


FeatureMatcher::FeatureMatcher(cv::Mat intrinsics_matrix, cv::Mat dist_coeffs,
                               bool use_modern_features, double focal_scale) :
  use_modern_features_(use_modern_features)
{
  intrinsics_matrix_ = intrinsics_matrix.clone();
  dist_coeffs_ = dist_coeffs.clone();
  new_intrinsics_matrix_ = intrinsics_matrix.clone();
  new_intrinsics_matrix_.at<double>(0,0) *= focal_scale;
  new_intrinsics_matrix_.at<double>(1,1) *= focal_scale;
}

cv::Mat FeatureMatcher::readUndistortedImage(const std::string& filename )
{
  cv::Mat img = cv::imread(filename), und_img, dbg_img;
  cv::undistort	(	img, und_img, intrinsics_matrix_, dist_coeffs_, new_intrinsics_matrix_ );

  return und_img;
}

void FeatureMatcher::extractFeatures()
{
  features_.resize(images_names_.size());
  descriptors_.resize(images_names_.size());
  feats_colors_.resize(images_names_.size());

  auto orb_detector = cv::ORB::create(10000, 1.2, 8);

  for( int i = 0; i < images_names_.size(); i++  )
  {
    std::cout<<"Computing descriptors for image "<<i<<std::endl;
    cv::Mat img = readUndistortedImage(images_names_[i]);


    //////////////////////////// Code to be completed (2/7) /////////////////////////////////
    // Extract salient points + descriptors from i-th image.
    //
    // A standard implementation (else branch) that uses the ORB features is already provided.
    // It stores them into the features_[i] and descriptors_[i] vectors, and extract the
    // color (cv::Vec3b) of each feature and store in feats_colors_[i] vector.
    //
    // You are required to implement an alternative, more modern feature detection and
    // description scheme inside the if (use_modern_features_) branch (e.g., by means the
    // loadExternalFeatures() function). Examples are SuperPoint
    // (https://github.com/eric-yyjau/pytorch-superpoint), DISK
    // (https://github.com/cvlab-epfl/disk) or ALIKED (https://github.com/Shiaoming/ALIKED),
    // or other alternatives.
    //
    // IMPORTANT: You also need to update the matching part, see the branch
    // if (use_modern_features_) in the FeatureMatcher::exhaustiveMatching() method.
    // For some methods, the feature description and matching phase are merged,
    // so you may only need to change Feature Matcher::exhaustive Matching()

    if (use_modern_features_)
    {
      // OPTION A: Inference inside C++
      // 1. Load a pre-trained model (e.g., SuperPoint.onnx) using cv::dnn::readNet().
      // 2. Convert 'img' to a blob and run net.forward().
      // 3. Post-process the output tensors to fill features_[i] and descriptors_[i].
      // See for example:
      // https://docs.opencv.org/4.x/dd/d55/pytorch_cls_c_tutorial_dnn_conversion.html
      // WARNING: By default, cv::dnn run in CPU only

      // OPTION B: Data Loading (Fallback)
      // If local hardware doesn't support inference, implement loadExternalFeatures()
      // to read keypoints and descriptors from a file (e.g., .txt) generated
      // beforehand by a Python script on your dataset.
      // loadExternalFeatures(image_path, features_[i], descriptors_[i]);

      // Remeber to Look-up features colors!

      //
      //keypoints already extracted in python implementation script
      //external features and matches will be loaded in exhaustiveMatching().
      //
    }
    else
    {
      // Standard ready to use ORB implementation
      orb_detector->detectAndCompute(img, cv::Mat(), features_[i], descriptors_[i]);

      // Look-up features colors
      feats_colors_[i].reserve(features_[i].size());
      for( auto &f : features_[i])
      {
        feats_colors_[i].emplace_back(img.at<cv::Vec3b>(f.pt.y, f.pt.x));
      }
    }
    /////////////////////////////////////////////////////////////////////////////////////////
  }
}

//HELPER function to deal with directories and files created by Python script
static boost::filesystem::path getSuperGlueRootFromImage(const std::string& image_name)
{
  
  boost::filesystem::path image_path(image_name);

  boost::filesystem::path images_folder = image_path.parent_path();

  boost::filesystem::path datasets_folder = images_folder.parent_path();

  std::string images_folder_name = images_folder.filename().string(); //for example images_1, images_2...

  return datasets_folder / ("superglue_" + images_folder_name); //directory contained superglue artifacts
}

//HELPER function to retrieve features obtained in the external Python script
static void loadExternalFeaturesForImage(int image_id, const std::vector<std::string>& images_names, std::vector<std::vector<cv::KeyPoint>>& features, std::vector<std::vector<cv::Vec3b>>& feats_colors)
{
  // If features for this image are already loaded, do nothing.
  if (!features[image_id].empty())
  {
    return;
  }

  boost::filesystem::path image_path(images_names[image_id]);

  boost::filesystem::path superglue_root = getSuperGlueRootFromImage(images_names[image_id]);

  std::string imgName = image_path.stem().string();

  boost::filesystem::path feature_file = superglue_root / "features" / (imgName + ".txt");

  std::ifstream file_input(feature_file.string());

  if (!file_input)
  {
    throw std::runtime_error(std::string("Cannot open SuperGlue feature file: ") + feature_file.string());
  }

  int num_features = 0;
  file_input >> num_features;  //first row of the file containes number of extracted features

  features[image_id].clear();
  feats_colors[image_id].clear();

  features[image_id].reserve(num_features);
  feats_colors[image_id].reserve(num_features);

  for (int k = 0; k < num_features; k++)  //process every row of the file
  {
    float x, y;
    int b, g, r;

    file_input >> x >> y >> b >> g >> r;

    features[image_id].emplace_back(cv::Point2f(x, y), 1.0f);

    feats_colors[image_id].emplace_back(static_cast<uchar>(b), static_cast<uchar>(g), static_cast<uchar>(r));
  }

  std::cout << "Loaded " << features[image_id].size() << " SuperGlue features for image " << image_id << std::endl;
}

//HELPER function to retrieve matches between img pairs obtained in the external Python script
static void loadExternalMatchesForPair(int image_i, int image_j, const std::vector<std::string>& images_names, const std::vector<std::vector<cv::KeyPoint>>& features, std::vector<cv::DMatch>& inlier_matches)
{
  boost::filesystem::path image_i_path(images_names[image_i]);
  boost::filesystem::path image_j_path(images_names[image_j]);

  boost::filesystem::path superglue_root = getSuperGlueRootFromImage(images_names[image_i]);

  std::string img_i = image_i_path.stem().string();
  std::string img_j = image_j_path.stem().string();

  //retrieve file with the matches
  boost::filesystem::path matches_file = superglue_root / "matches" / (img_i + "_" + img_j + ".txt");

  std::ifstream file_input(matches_file.string());

  if (!file_input)
  {
    throw std::runtime_error(std::string("Cannot open SuperGlue matches file: ") + matches_file.string());
  }

  int num_matches = 0;
  file_input >> num_matches;

  inlier_matches.clear();
  inlier_matches.reserve(num_matches);

  for (int k = 0; k < num_matches; k++)
  {
    int idx0, idx1;
    float score;

    file_input >> idx0 >> idx1 >> score;

    // idx0 refers to features[image_i]
    // idx1 refers to features[image_j]
    if (idx0 >= 0 && idx0 < static_cast<int>(features[image_i].size()) && idx1 >= 0 && idx1 < static_cast<int>(features[image_j].size()))
    {
      cv::DMatch match;

      match.queryIdx = idx0;
      match.trainIdx = idx1;

      // SuperGlue gives confidence: higher is better.
      // OpenCV DMatch stores distance: lower is better.
      match.distance = 1.0f - score;

      inlier_matches.emplace_back(match);
    }
  }

  std::cout << "Loaded " << inlier_matches.size() << " SuperGlue matches for images " << image_i << " and " << image_j << std::endl;
}

void FeatureMatcher::exhaustiveMatching()
{
  std::vector<cv::DMatch> matches, inlier_matches;
  
  for( int i = 0; i < images_names_.size() - 1; i++ )
  {
    for( int j = i + 1; j < images_names_.size(); j++ )
    {
      std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;
      std::vector<cv::DMatch> matches, inlier_matches;

      if( use_modern_features_ )
      {
        // Modern descriptors (SuperPoint, etc.) are usually float matrices.
        // You may use a BruteForce with L2 distance and enable cross-check for better precision.
        //
        // You could also use a modern Matching Network such as SuperGlue/LightGlue
        // (https://github.com/magicleap/supergluepretrainednetwork) subsequent
        // geometric verification (Code to be completed (1/7)) is not required,
        // since these networks perform both matching and geometric verification.
        // In this case, you may follow OPTION A or OPTION A (see above).
        
        /////////////////////////////////////////////////////////////////////////////////////////
        loadExternalFeaturesForImage(i, images_names_, features_, feats_colors_);

        loadExternalFeaturesForImage(j, images_names_, features_, feats_colors_);

        loadExternalMatchesForPair(i, j, images_names_, features_, inlier_matches);

        //Superglue already filtered matches, consider them as final result
        if (inlier_matches.size() > 5)
        {
          setMatches( i, j, inlier_matches);
        }

        continue;

        /////////////////////////////////////////////////////////////////////////////////////////

      }
      else
      {
        std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;
        auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        matcher->match(descriptors_[i], descriptors_[j], matches);
      }

      //////////////////////////// Code to be completed (1/7) /////////////////////////////////
      // Perform Geometric Verification of matches, possibly discarding the outliers
      // (remember that features have been extracted from undistorted images that now has
      // new_intrinsics_matrix_ as K matrix and no distortions).
      // As geometric models, use both the Essential matrix and the Homograph matrix,
      // both by setting new_intrinsics_matrix_ as K matrix.
      // As threshold in the functions to estimate both models, you may use 1.0 or similar.
      // Store inlier matches into the inlier_matches vector
      // Do not set matches between two images if the amount of inliers matches
      // (i.e., geomatrically verified matches) is small (say <= 5 matches)
      // In case of success, set the matches with the function:
      //
      // setMatches( i, j, inlier_matches);
      //
      // where i,j matched images indices.
      /////////////////////////////////////////////////////////////////////////////////////////
      
      //conversion from DMatch to 2d point type to work with opencv E and H estimation methods
      std::vector<cv::Point2f> points_i, points_j;
      
      for (const auto &m : matches)
      {
        points_i.emplace_back(features_[i][m.queryIdx].pt);
        points_j.emplace_back(features_[j][m.trainIdx].pt);
      }

      cv::Mat essential_mask, homography_mask;
      const double threshold = 1.0;
      
      //estimate E and H
      cv::Mat E = cv::findEssentialMat(points_i, points_j, new_intrinsics_matrix_, cv::RANSAC, 0.999, threshold, essential_mask);
      cv::Mat H = cv::findHomography(points_i, points_j, cv::RANSAC, threshold, homography_mask);

      int essential_inlier_value = cv::countNonZero(essential_mask);
      int homography_inlier_value = cv::countNonZero(homography_mask);

      //choose best mask between the 2 transformations
      cv::Mat best_mask;
      if (essential_inlier_value >= homography_inlier_value)
      {
        best_mask = essential_mask;
      }
      else
      {
        best_mask = homography_mask;
      }

      //insert in inlier array
      for (int k=0; k < static_cast<int>(matches.size()); k++)
      {
        if (best_mask.at<uchar>(k))
        {
          inlier_matches.emplace_back(matches[k]);
        }
      }

      if (inlier_matches.size() > 5)
      {
        setMatches( i, j, inlier_matches);
      }

      
      /////////////////////////////////////////////////////////////////////////////////////////
    }
  }
}

void FeatureMatcher::writeToFile ( const std::string& filename, bool normalize_points ) const
{
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    std::cerr << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_poses_, num_points_, num_observations_);

  double *tmp_observations;
  cv::Mat dst_pts;
  if(normalize_points)
  {
    cv::Mat src_obs( num_observations_,1, cv::traits::Type<cv::Vec2d>::value,
                     const_cast<double *>(observations_.data()));
    cv::undistortPoints(src_obs, dst_pts, new_intrinsics_matrix_, cv::Mat());
    tmp_observations = reinterpret_cast<double *>(dst_pts.data);
  }
  else
  {
    tmp_observations = const_cast<double *>(observations_.data());
  }

  for (int i = 0; i < num_observations_; ++i)
  {
    fprintf(fptr, "%d %d", pose_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", tmp_observations[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  if( colors_.size() == 3*num_points_ )
  {
    for (int i = 0; i < num_points_; ++i)
      fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
  }

  fclose(fptr);
}

void FeatureMatcher::testMatches( double scale )
{
  // For each pose, prepare a map that reports the pairs [point index, observation index]
  std::vector< std::map<int,int> > cam_observation( num_poses_ );
  for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
  {
    int i_cam = pose_index_[i_obs], i_pt = point_index_[i_obs];
    cam_observation[i_cam][i_pt] = i_obs;
  }

  for( int r = 0; r < num_poses_; r++ )
  {
    for (int c = r + 1; c < num_poses_; c++)
    {
      int num_mathces = 0;
      std::vector<cv::DMatch> matches;
      std::vector<cv::KeyPoint> features0, features1;
      for (auto const &co_iter: cam_observation[r])
      {
        if (cam_observation[c].find(co_iter.first) != cam_observation[c].end())
        {
          features0.emplace_back(observations_[2*co_iter.second],observations_[2*co_iter.second + 1], 0.0);
          features1.emplace_back(observations_[2*cam_observation[c][co_iter.first]],observations_[2*cam_observation[c][co_iter.first] + 1], 0.0);
          matches.emplace_back(num_mathces,num_mathces, 0);
          num_mathces++;
        }
      }
      cv::Mat img0 = readUndistortedImage(images_names_[r]),
          img1 = readUndistortedImage(images_names_[c]),
          dbg_img;

      cv::drawMatches(img0, features0, img1, features1, matches, dbg_img);
      cv::resize(dbg_img, dbg_img, cv::Size(), scale, scale);
      cv::imshow("", dbg_img);
      if (cv::waitKey() == 27)
        return;
    }
  }
}

void FeatureMatcher::setMatches( int pos0_id, int pos1_id, const std::vector<cv::DMatch> &matches )
{

  const auto &features0 = features_[pos0_id];
  const auto &features1 = features_[pos1_id];

  auto pos_iter0 = pose_id_map_.find(pos0_id),
      pos_iter1 = pose_id_map_.find(pos1_id);

  // Already included position?
  if( pos_iter0 == pose_id_map_.end() )
  {
    pose_id_map_[pos0_id] = num_poses_;
    pos0_id = num_poses_++;
  }
  else
    pos0_id = pose_id_map_[pos0_id];

  // Already included position?
  if( pos_iter1 == pose_id_map_.end() )
  {
    pose_id_map_[pos1_id] = num_poses_;
    pos1_id = num_poses_++;
  }
  else
    pos1_id = pose_id_map_[pos1_id];

  for( auto &match:matches)
  {

    // Already included observations?
    uint64_t obs_id0 = poseFeatPairID(pos0_id, match.queryIdx ),
        obs_id1 = poseFeatPairID(pos1_id, match.trainIdx );
    auto pt_iter0 = point_id_map_.find(obs_id0),
        pt_iter1 = point_id_map_.find(obs_id1);
    // New point
    if( pt_iter0 == point_id_map_.end() && pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = num_points_++;
      point_id_map_[obs_id0] = point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);

      // Average color between two corresponding features (suboptimal since we shouls also consider
      // the other observations of the same point in the other images)
      cv::Vec3f color = (cv::Vec3f(feats_colors_[pos0_id][match.queryIdx]) +
                        cv::Vec3f(feats_colors_[pos1_id][match.trainIdx]))/2;

      colors_.push_back(cvRound(color[2]));
      colors_.push_back(cvRound(color[1]));
      colors_.push_back(cvRound(color[0]));

      num_observations_++;
      num_observations_++;
    }
      // New observation
    else if( pt_iter0 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id1];
      point_id_map_[obs_id0] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      num_observations_++;
    }
    else if( pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id0];
      point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);
      num_observations_++;
    }
//    else if( pt_iter0->second != pt_iter1->second )
//    {
//      std::cerr<<"Shared observations does not share 3D point!"<<std::endl;
//    }
  }
}
void FeatureMatcher::reset()
{
  point_index_.clear();
  pose_index_.clear();
  observations_.clear();
  colors_.clear();

  num_poses_ = num_points_ = num_observations_ = 0;
}
