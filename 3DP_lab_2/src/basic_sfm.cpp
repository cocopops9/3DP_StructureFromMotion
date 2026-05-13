#include "basic_sfm.h"

#include <iostream>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

struct ReprojectionError
{
  //////////////////////////// Code to be completed (5/7) //////////////////////////////////
  // This class should include an auto-differentiable cost function (see Ceres Solver docs).
  // Remember that we are dealing with a normalized, canonical camera:
  // point projection is easy! To rotete a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  //////////////////////////////////////////////////////////////////////////////////////////

  double observed_x;
  double observed_y;

  ReprojectionError(double obs_x, double obs_y)
      : observed_x(obs_x), observed_y(obs_y) {}

  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const
  {
    // camera[0..2] is the axis-angle rotation; rotate the 3D point.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3..5] is the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Canonical camera (K = identity, no distortion): project by dividing by depth.
    T predicted_x = p[0] / p[2];
    T predicted_y = p[1] / p[2];

    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to build an AutoDiffCostFunction. Template parameters order:
  // <Functor, residual_dim, camera_block_size, point_block_size> = <_, 2, 6, 3>.
  static ceres::CostFunction* Create(double obs_x, double obs_y)
  {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
        new ReprojectionError(obs_x, obs_y));
  }

  /////////////////////////////////////////////////////////////////////////////////////////
};


namespace
{
typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template<typename T>
void FscanfOrDie(FILE* fptr, const char* format, T* value)
{
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1)
  {
    cerr << "Invalid UW data file.";
    exit(-1);
  }
}

}  // namespace


BasicSfM::~BasicSfM()
{
  reset();
}

void BasicSfM::reset()
{
  point_index_.clear();
  cam_pose_index_.clear();
  observations_.clear();
  colors_.clear();
  parameters_.clear();

  num_cam_poses_ = num_points_ = num_observations_ = num_parameters_ = 0;
}

void BasicSfM::readFromFile ( const std::string& filename, bool load_initial_guess, bool load_colors  )
{
  reset();

  FILE* fptr = fopen(filename.c_str(), "r");

  if (fptr == NULL)
  {
    cerr << "Error: unable to open file " << filename;
    return;
  };

  // This wil die horribly on invalid files. Them's the breaks.
  FscanfOrDie(fptr, "%d", &num_cam_poses_);
  FscanfOrDie(fptr, "%d", &num_points_);
  FscanfOrDie(fptr, "%d", &num_observations_);

  cout << "Header: " << num_cam_poses_
       << " " << num_points_
       << " " << num_observations_<<std::endl;

  point_index_.resize(num_observations_);
  cam_pose_index_.resize(num_observations_);
  observations_.resize(2 * num_observations_);

  num_parameters_ = camera_block_size_ * num_cam_poses_ + point_block_size_ * num_points_;
  parameters_.resize(num_parameters_);

  for (int i = 0; i < num_observations_; ++i)
  {
    FscanfOrDie(fptr, "%d", cam_pose_index_.data() + i);
    FscanfOrDie(fptr, "%d", point_index_.data() + i);
    for (int j = 0; j < 2; ++j)
    {
      FscanfOrDie(fptr, "%lf", observations_.data() + 2*i + j);
    }
  }

  if( load_colors )
  {
    colors_.resize(3*num_points_);
    for (int i = 0; i < num_points_; ++i)
    {
      int r,g,b;
      FscanfOrDie(fptr, "%d", &r );
      FscanfOrDie(fptr, "%d", &g);
      FscanfOrDie(fptr, "%d", &b );
      colors_[i*3] = r;
      colors_[i*3 + 1] = g;
      colors_[i*3 + 2] = b;
    }
  }

  if( load_initial_guess )
  {
    cam_pose_optim_iter_.resize(num_cam_poses_, 1 );
    pts_optim_iter_.resize( num_points_, 1 );

    for (int i = 0; i < num_parameters_; ++i)
    {
      FscanfOrDie(fptr, "%lf", parameters_.data() + i);
    }
  }
  else
  {
    memset(parameters_.data(), 0, num_parameters_*sizeof(double));
    // Masks used to indicate which cameras and points have been optimized so far
    cam_pose_optim_iter_.resize(num_cam_poses_, 0 );
    pts_optim_iter_.resize( num_points_, 0 );
    
  }

  fclose(fptr);
}


void BasicSfM::writeToFile (const string& filename, bool write_unoptimized ) const
{
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    cerr << "Error: unable to open file " << filename;
    return;
  };

  if( write_unoptimized )
  {
    fprintf(fptr, "%d %d %d\n", num_cam_poses_, num_points_, num_observations_);

    for (int i = 0; i < num_observations_; ++i)
    {
      fprintf(fptr, "%d %d", cam_pose_index_[i], point_index_[i]);
      for (int j = 0; j < 2; ++j) {
        fprintf(fptr, " %g", observations_[2 * i + j]);
      }
      fprintf(fptr, "\n");
    }

    if( colors_.size() == num_points_*3 )
    {
      for (int i = 0; i < num_points_; ++i)
        fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
    }

    for (int i = 0; i < num_cam_poses_; ++i)
    {
      const double *camera = parameters_.data() + camera_block_size_ * i;
      for (int j = 0; j < camera_block_size_; ++j) {
        fprintf(fptr, "%.16g\n", camera[j]);
      }
    }

    const double* points = pointBlockPtr();
    for (int i = 0; i < num_points_; ++i)
    {
      const double* point = points + i * point_block_size_;
      for (int j = 0; j < point_block_size_; ++j) {
        fprintf(fptr, "%.16g\n", point[j]);
      }
    }
  }
  else
  {
    int num_cameras = 0, num_points = 0, num_observations = 0;

    for (int i = 0; i < num_cam_poses_; ++i)
      if( cam_pose_optim_iter_[i] > 0 ) num_cameras++;

    for (int i = 0; i < num_points_; ++i)
      if( pts_optim_iter_[i] > 0 ) num_points++;

    for (int i = 0; i < num_observations_; ++i)
      if( cam_pose_optim_iter_[cam_pose_index_[i]] > 0  && pts_optim_iter_[point_index_[i]] > 0 ) num_observations++;

    fprintf(fptr, "%d %d %d\n", num_cameras, num_points, num_observations);

    for (int i = 0; i < num_observations_; ++i)
    {
      if( cam_pose_optim_iter_[cam_pose_index_[i]] > 0  && pts_optim_iter_[point_index_[i]] > 0 )
      {
        fprintf(fptr, "%d %d", cam_pose_index_[i], point_index_[i]);
        for (int j = 0; j < 2; ++j) {
          fprintf(fptr, " %g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
      }
    }

    if( colors_.size() == num_points_*3 )
    {
      for (int i = 0; i < num_points_; ++i)
      {
        if(pts_optim_iter_[i] > 0)
          fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
      }
    }

    for (int i = 0; i < num_cam_poses_; ++i)
    {
      if( cam_pose_optim_iter_[i] > 0 )
      {
        const double *camera = parameters_.data() + camera_block_size_ * i;
        for (int j = 0; j < camera_block_size_; ++j)
        {
          fprintf(fptr, "%.16g\n", camera[j]);
        }
      }
    }

    const double* points = pointBlockPtr();
    for (int i = 0; i < num_points_; ++i)
    {
      if( pts_optim_iter_[i] > 0 )
      {
        const double* point = points + i * point_block_size_;
        for (int j = 0; j < point_block_size_; ++j)
        {
          fprintf(fptr, "%.16g\n", point[j]);
        }
      }
    }
  }

  fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare.
void BasicSfM::writeToPLYFile (const string& filename, bool write_unoptimized ) const
{
  ofstream of(filename.c_str());

  int num_cameras, num_points;

  if( write_unoptimized )
  {
    num_cameras = num_cam_poses_;
    num_points = num_points_;
  }
  else
  {
    num_cameras = 0;
    num_points = 0;
    for (int i = 0; i < num_cam_poses_; ++i)
      if( cam_pose_optim_iter_[i] > 0 ) num_cameras++;

    for (int i = 0; i < num_points_; ++i)
      if( pts_optim_iter_[i] > 0 ) num_points++;
  }

  of << "ply"
     << '\n' << "format ascii 1.0"
     << '\n' << "element vertex " << num_cameras + num_points
     << '\n' << "property float x"
     << '\n' << "property float y"
     << '\n' << "property float z"
     << '\n' << "property uchar red"
     << '\n' << "property uchar green"
     << '\n' << "property uchar blue"
     << '\n' << "end_header" << endl;

  bool write_colors = ( colors_.size() == num_points_*3 );
  if( write_unoptimized )
  {
    // Export extrinsic data (i.e. camera centers) as green points.
    double center[3];
    for (int i = 0; i < num_cam_poses_; ++i)
    {
      const double* camera = cameraBlockPtr(i);
      cam2center (camera, center);
      of << center[0] << ' ' << center[1] << ' ' << center[2]
         << " 0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points.
    const double* points = pointBlockPtr();
    for (int i = 0; i < num_points_; ++i)
    {
      const double* point = points + i * point_block_size_;
      for (int j = 0; j < point_block_size_; ++j)
      {
        of << point[j] << ' ';
      }
      if (write_colors )
        of << int(colors_[3*i])<<" " << int(colors_[3*i + 1])<<" "<< int(colors_[3*i + 2])<<"\n";
      else
        of << "255 255 255\n";
    }
  }
  else
  {
    // Export extrinsic data (i.e. camera centers) as green points.
    double center[3];
    for (int i = 0; i < num_cam_poses_; ++i)
    {
      if( cam_pose_optim_iter_[i] > 0 )
      {
        const double* camera = cameraBlockPtr(i);
        cam2center (camera, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 0 255 0" << '\n';
      }
    }

    // Export the structure (i.e. 3D Points) as white points.
    const double* points = pointBlockPtr();;
    for (int i = 0; i < num_points_; ++i)
    {
      if( pts_optim_iter_[i] > 0 )
      {
        const double* point = points + i * point_block_size_;
        for (int j = 0; j < point_block_size_; ++j)
        {
          of << point[j] << ' ';
        }
        if (write_colors )
          of << int(colors_[3*i])<<" " << int(colors_[3*i + 1])<<" "<< int(colors_[3*i + 2])<<"\n";
        else
          of << "255 255 255\n";
      }
    }
  }
  of.close();
}

/* c_{w,cam} = R_{cam}'*[0 0 0]' - R_{cam}'*t_{cam} -> c_{w,cam} = - R_{cam}'*t_{cam} */
void BasicSfM::cam2center (const double* camera, double* center) const
{
  ConstVectorRef angle_axis_ref(camera, 3);

  Eigen::VectorXd inverse_rotation = -angle_axis_ref;
  ceres::AngleAxisRotatePoint(inverse_rotation.data(), camera + 3, center);
  VectorRef(center, 3) *= -1.0;
}

/* [0 0 0]' = R_{cam}*c_{w,cam} + t_{cam} -> t_{cam} = - R_{cam}*c_{w,cam} */
void BasicSfM::center2cam (const double* center, double* camera) const
{
  ceres::AngleAxisRotatePoint(camera, center, camera + 3);
  VectorRef(camera + 3, 3) *= -1.0;
}


bool BasicSfM::checkCheiralityConstraint (int pos_idx, int pt_idx )
{
  double *camera = cameraBlockPtr(pos_idx),
         *point = pointBlockPtr(pt_idx);

  double p[3];
  ceres::AngleAxisRotatePoint(camera, point, p);

  // camera[5] is the z cooordinate wrt the camera at pose pose_idx
  p[2] += camera[5];
  return p[2] > 0;
}

void BasicSfM::printPose ( int idx )  const
{
  const double *cam = cameraBlockPtr(idx);
  std::cout<<"camera["<<idx<<"]"<<std::endl
           <<"{"<<std::endl
           <<"\t r_vec : ("<<cam[0]<<", "<<cam[1]<<", "<<cam[2]<<")"<<std::endl
           <<"\t t_vec : ("<<cam[3]<<", "<<cam[4]<<", "<<cam[5]<<")"<<std::endl;

  std::cout<<"}"<<std::endl;
}


void BasicSfM::printPointParams ( int idx ) const
{
  const double *pt = pointBlockPtr(idx);
  std::cout<<"point["<<idx<<"] : ("<<pt[0]<<", "<<pt[1]<<", "<<pt[2]<<")"<<std::endl;
}


void BasicSfM::solve()
{
  // For each camera pose, prepare a map that reports the pairs [point index, observation index]
  // This map is used to quickly retrieve the observation index given a 3D point index
  // For instance, to query if the camera pose with index i_cam observed the
  // 3D point with index i_pt, check if cam_observation_[i_cam].find( i_pt ) is not cam_observation_[i_cam].end(), i.e.,:
  // if(cam_observation_[i_cam].find( i_pt ) != cam_observation_[i_cam].end())  { .... }
  // In case of success, you can retrieve the observation index obs_id simply with:
  // obs_id = cam_observation_[i_cam][i_pt]
  cam_observation_ = vector< map<int,int> > (num_cam_poses_ );
  for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
  {
    int i_cam = cam_pose_index_[i_obs], i_pt = point_index_[i_obs];
    cam_observation_[i_cam][i_pt] = i_obs;
  }

  // Compute a (symmetric) num_cam_poses_ X num_cam_poses_ matrix
  // that counts the number of correspondences between pairs of camera poses
  Eigen::MatrixXi corr = Eigen::MatrixXi::Zero(num_cam_poses_, num_cam_poses_);

  for(int r = 0; r < num_cam_poses_; r++ )
  {
    for(int c = r + 1; c < num_cam_poses_; c++ )
    {
      int nc = 0;
      for( auto const& co_iter : cam_observation_[r] )
      {
        if( cam_observation_[c].find(co_iter.first ) != cam_observation_[c].end() )
          nc++;
      }
      corr(r,c) = nc;
    }
  }

  // num_cam_poses_ X num_cam_poses_ matrix to mask already tested seed pairs
  // already_tested_pair(r,c) == 0 -> not tested pair
  // already_tested_pair(r,c) != 0 -> already tested pair
  Eigen::MatrixXi already_tested_pair = Eigen::MatrixXi::Zero(num_cam_poses_, num_cam_poses_);


  // Indices of the two camera poses that define the initial seed pair
  int seed_pair_idx0, seed_pair_idx1;

  
  const bool use_opt1 = (std::getenv("SFM_opt1") != nullptr &&
                          std::string(std::getenv("SFM_opt1")) == "1");

  // Look for a suitable seed pair (basic strategy)....
  if (!use_opt1) while( true )
  {
    int max_corr = -1;
    for(int r = 0; r < num_cam_poses_; r++ )
    {
      for (int c = r + 1; c < num_cam_poses_; c++)
      {
        if( !already_tested_pair(r,c) && corr(r,c) > max_corr )
        {
          max_corr = corr(r,c);
          seed_pair_idx0 = r;
          seed_pair_idx1 = c;
        }
      }
    }

    if( max_corr < 0 )
    {
      std::cout<<"No seed pair found, exiting"<<std::endl;
      return;
    }
    already_tested_pair(seed_pair_idx0, seed_pair_idx1) = 1;

    if (incrementalReconstruction( seed_pair_idx0, seed_pair_idx1 ))
    {
      std::cout<<"Recostruction completed, exiting"<<std::endl;
      return;
    }
    else
    {
      std::cout<<"Try to look for a better seed pair"<<std::endl;
    }
  }
  

  //////////////////////////// Code to be completed (OPTIONAL 1) ////////////////////////////////
  // Implement an alternative seed selection strategy. Just comment the basic seed selection
  // strategy implemented above and replace it with yours.
  //////////////////////////////////////////////////////////////////////////////////////////////
  if (use_opt1)
  {
    struct SeedCandidate{
        int cam0 = -1;
        int cam1 = -1;

        int common_observations = 0;
        int essential_inliers = 0;
        int homography_inliers = 0;
        int pose_inliers = 0;

        double lateral_motion = 0.0;
        double forward_motion = 0.0;
        double score = -1.0;
    };

    std::vector<SeedCandidate> seed_candidates;

    const double ransac_threshold = 0.001;
    const int min_common_observations = 8;
    const int min_pose_inliers = 8;

     // Canonical camera: observations are already normalized.
    cv::Mat_<double> intrinsics_matrix = cv::Mat_<double>::eye(3, 3);

    for (int r = 0; r < num_cam_poses_; r++) {
        for (int c = r + 1; c < num_cam_poses_; c++) {
          
            if (corr(r, c) < min_common_observations)
            continue;

            std::vector<cv::Point2d> points0, points1;
            points0.reserve(corr(r, c));
            points1.reserve(corr(r, c));

            // Collect common observations between camera r and camera c.
            for (auto const& co_iter : cam_observation_[r]) {
                
                const int pt_idx = co_iter.first;

                auto obs_c = cam_observation_[c].find(pt_idx);

                if (obs_c != cam_observation_[c].end()) {

                  const int obs_r_idx = co_iter.second;
                  const int obs_c_idx = obs_c->second;

                  points0.emplace_back(observations_[2 * obs_r_idx], observations_[2 * obs_r_idx + 1]);
                  points1.emplace_back(observations_[2 * obs_c_idx], observations_[2 * obs_c_idx + 1]);
                }
            }

            if (points0.size() < min_common_observations)
                continue;

            cv::Mat inlier_mask_E, inlier_mask_H;
            cv::Mat E = cv::findEssentialMat(points0, points1, intrinsics_matrix, cv::RANSAC, 0.999, ransac_threshold, inlier_mask_E);

            if (E.empty() || inlier_mask_E.empty())
              continue;

            cv::Mat H = cv::findHomography(points0, points1, cv::RANSAC, ransac_threshold, inlier_mask_H);

            const int inliers_E = cv::countNonZero(inlier_mask_E);
            const int inliers_H = (!H.empty() && !inlier_mask_H.empty()) ? cv::countNonZero(inlier_mask_H) : 0;

            // If H explains the pair almost as well as E, the pair is likely planar
            // or close to a pure rotation, so it is not ideal for SfM initialization.
            if (inliers_E <= inliers_H)
              continue;

            cv::Mat R, t;
            cv::Mat recover_mask = inlier_mask_E.clone();

            const int pose_inliers = cv::recoverPose(E, points0, points1, intrinsics_matrix, R, t, recover_mask);

            if (pose_inliers < min_pose_inliers)
              continue;

            const double tx = std::fabs(t.at<double>(0, 0));
            const double ty = std::fabs(t.at<double>(1, 0));
            const double tz = std::fabs(t.at<double>(2, 0));
            const double lateral_motion = std::sqrt(tx * tx + ty * ty);
            const double forward_motion = std::fabs(tz);

            if (lateral_motion < forward_motion)
              continue;

            SeedCandidate candidate;
            candidate.cam0 = r;
            candidate.cam1 = c;
            candidate.common_observations = static_cast<int>(points0.size());
            candidate.essential_inliers = inliers_E;
            candidate.homography_inliers = inliers_H;
            candidate.pose_inliers = pose_inliers;
            candidate.lateral_motion = lateral_motion;
            candidate.forward_motion = forward_motion;      

            // Score:   
            candidate.score =
                2.0 * static_cast<double>(pose_inliers)
                + 2.0 * static_cast<double>(candidate.common_observations)
                + 50.0 * static_cast<double>(inliers_E - inliers_H)
                + 8000.0 * lateral_motion
                - 2500.0 * forward_motion; 

            seed_candidates.push_back(candidate);
        }
    }

    std::sort(seed_candidates.begin(), seed_candidates.end(), [](const SeedCandidate& a, const SeedCandidate& b) {return a.score > b.score;});

    if (seed_candidates.empty()) {
        std::cout << "No seed pair found with the alternative seed selection, exiting" << std::endl;
        return;
    }

    std::cout << "Alternative seed selection candidates:" << std::endl;
    const int num_to_print = std::min<int>(10, seed_candidates.size());

    for (int i = 0; i < num_to_print; i++) {
        
        const SeedCandidate& candidate = seed_candidates[i];

        std::cout << "  rank " << i + 1
                  << " pair (" << candidate.cam0 << ", " << candidate.cam1 << ")"
                  << " common=" << candidate.common_observations
                  << " E=" << candidate.essential_inliers
                  << " H=" << candidate.homography_inliers
                  << " pose=" << candidate.pose_inliers
                  << " lateral=" << candidate.lateral_motion
                  << " forward=" << candidate.forward_motion
                  << " score=" << candidate.score << std::endl;
    }

    for (const SeedCandidate& candidate : seed_candidates) {
        std::cout << "Trying alternative seed pair ("
                  << candidate.cam0 << ", " << candidate.cam1 << ")"
                  << " score=" << candidate.score << std::endl;

        if (incrementalReconstruction(candidate.cam0, candidate.cam1)) {
            std::cout << "Recostruction completed, exiting" << std::endl;
            return;
        }

        std::cout << "Alternative seed pair failed, trying next candidate" << std::endl;
    }

    std::cout << "No valid reconstruction found with the alternative seed selection, exiting" << std::endl;
    return;
  }
  ////////////////////////////////////////////////////////////////////////////////////////
}

bool BasicSfM::incrementalReconstruction( int seed_pair_idx0, int seed_pair_idx1 )
{
  // Reset all parameters: we are starting a brand new reconstruction from a new seed pair
  memset(parameters_.data(), 0, num_parameters_*sizeof(double));
  // Masks used to indicate which cameras and points have been optimized so far
  cam_pose_optim_iter_.assign(num_cam_poses_, 0 );
  pts_optim_iter_.assign( num_points_, 0 );

  // Init R,t between the seed pair
  cv::Mat init_r_mat, init_r_vec, init_t_vec;

  std::vector<cv::Point2d> points0, points1;
  cv::Mat inlier_mask_E, inlier_mask_H;

  // Collect matches between the two images of the seed pair, to be used to extract the models E and H
  for (auto const &co_iter: cam_observation_[seed_pair_idx0])
  {
    if (cam_observation_[seed_pair_idx1].find(co_iter.first) != cam_observation_[seed_pair_idx1].end())
    {
      points0.emplace_back(observations_[2*co_iter.second],observations_[2*co_iter.second + 1]);
      points1.emplace_back(observations_[2*cam_observation_[seed_pair_idx1][co_iter.first]],
                           observations_[2*cam_observation_[seed_pair_idx1][co_iter.first] + 1]);
    }
  }

  // Canonical camera so identity K
  cv::Mat_<double> intrinsics_matrix = cv::Mat_<double>::eye(3,3);

  //////////////////////////// Code to be completed (3/7) /////////////////////////////////
  // Extract both Essential matrix E and Homograph matrix H.
  // As threshold in the functions to estimate both models, you may use 0.001 or similar.
  // Check that the number of inliers for the model E is higher than the number of
  // inliers for the model H (-> use inlier_mask_E and inlier_mask_H defined above <-),
  // otherwise, return false  (we will try a new seed pair),
  // If true, recover from E the initial rigid body transformation between seed_pair_idx0
  // and seed_pair_idx1 by using the cv::recoverPose() OpenCV function
  // (use inlier_mask_E as input/output param).
  // Check if the recovered transformation is mainly given by a sideward motion,
  // which is better than forward one.
  // Otherwise, return false (we will try a new seed pair)
  // In case of "good" sideward motion, store the transformation into init_r_mat and
  // init_t_vec; defined above
  /////////////////////////////////////////////////////////////////////////////////////////

  // Observations are in canonical coordinates; intrinsics_matrix is identity here, so the
  // RANSAC threshold is in normalized-pixel units. 0.001 corresponds to ~1 pixel.
  cv::Mat E = cv::findEssentialMat(points0, points1, intrinsics_matrix,
                                   cv::RANSAC, 0.999, 0.001, inlier_mask_E);
  cv::Mat H = cv::findHomography(points0, points1, cv::RANSAC, 0.001, inlier_mask_H);

  int inliers_E = cv::countNonZero(inlier_mask_E);
  int inliers_H = cv::countNonZero(inlier_mask_H);

  std::cout << "Seed pair (" << seed_pair_idx0 << ", " << seed_pair_idx1
            << "): inliers E=" << inliers_E << " H=" << inliers_H << std::endl;

  // A homography explaining at least as many matches as the Essential matrix indicates a
  // planar or purely rotational configuration. Reject such seeds.
  if (inliers_E <= inliers_H)
    return false;

  // Recover the relative pose from E. init_r_mat and init_t_vec hold the rotation and
  // unit-norm translation that map points from seed_pair_idx0 frame to seed_pair_idx1
  // frame. inlier_mask_E is refined by recoverPose.
  int valid_seed_points = cv::recoverPose(E, points0, points1, intrinsics_matrix, init_r_mat, init_t_vec, inlier_mask_E);
                  
  // Se la triangolazione iniziale produce pochi punti affidabili, scartiamo la coppia.
  if (valid_seed_points < 50) {
      std::cout << "Seed pair rejected: poor triangulation, only " 
                << valid_seed_points << " valid points." << std::endl;
      return false; 
  }

  // Since t is unit-norm, its components are direction cosines. A motion is "mainly
  // sideward" when at least one of |tx|, |ty| is >= 0.5. Reject pairs where both
  // lateral components are below 0.5 (forward-dominant motion, which yields poor
  // triangulation baselines).
  double tx = init_t_vec.at<double>(0,0);
  double ty = init_t_vec.at<double>(1,0);
  double tz = init_t_vec.at<double>(2,0);
  if (std::sqrt(tx*tx + ty*ty) < std::fabs(tz)) {
    std::cout << "Forward-dominant motion detected, rejecting." << std::endl;
    return false;
  }

  /////////////////////////////////////////////////////////////////////////////////////////

  int ref_cam_pose_idx = seed_pair_idx0, new_cam_pose_idx = seed_pair_idx1;

  // Initialize the first optimized poses, by integrating them into the registration
  // cam_pose_optim_iter_ and pts_optim_iter_ are simple mask vectors that define which camera poses and
  // which point positions have been already registered, specifically
  // if cam_pose_optim_iter_[pos_id] or pts_optim_iter_[id] are:
  // > 0 ---> The corresponding pose or point position has been already been estimated
  // == 0 ---> The corresponding pose or point position has not yet been estimated
  // == -1 ---> The corresponding pose or point position has been rejected due to e.g. outliers, etc...
  cam_pose_optim_iter_[ref_cam_pose_idx] = cam_pose_optim_iter_[new_cam_pose_idx] = 1;

  //Initialize the first RT wrt the reference position
  cv::Mat r_vec;
  // Recover the axis-angle rotation from init_r_mat
  cv::Rodrigues(init_r_mat, r_vec);

  // And update the parameters_ vector
  // First camera pose of the seed pair: just the identity transformation for now
  cv::Mat_<double> ref_rt_vec = (cv::Mat_<double>(3,1) << 0,0,0);
  initCamParams(ref_cam_pose_idx, ref_rt_vec, ref_rt_vec );
  // Second camera pose of the seed pair: the just recovered transformation
  initCamParams(new_cam_pose_idx, r_vec, init_t_vec );

  printPose(ref_cam_pose_idx);
  printPose(new_cam_pose_idx);

  // Triangulate the 3D points observed by both cameras
  cv::Mat_<double> proj_mat0 = cv::Mat_<double>::zeros(3, 4), proj_mat1(3, 4), hpoints4D;
  // First camera pose of the seed pair: just the identity transformation for now
  proj_mat0(0,0) = proj_mat0(1,1) = proj_mat0(2,2) = 1.0;
  // Second camera pose of the seed pair: the just recovered transformation
  init_r_mat.copyTo(proj_mat1(cv::Rect(0, 0, 3, 3)));
  init_t_vec.copyTo(proj_mat1(cv::Rect(3, 0, 1, 3)));

  cv::triangulatePoints(	proj_mat0, proj_mat1, points0, points1, hpoints4D );

  int match_idx = 0;

// Initialize the first optimized points.
// match_idx indexes ONLY the compact vectors of common matches:
// points0, points1, inlier_mask_E and hpoints4D.
for (auto const& co_iter : cam_observation_[ref_cam_pose_idx])
{
  const int pt_idx = co_iter.first;

  if (cam_observation_[new_cam_pose_idx].find(pt_idx) ==
      cam_observation_[new_cam_pose_idx].end())
  {
    continue;
  }

  if (match_idx >= hpoints4D.cols ||
      match_idx >= static_cast<int>(points0.size()))
  {
    std::cerr << "Internal indexing error during seed triangulation" << std::endl;
    return false;
  }

  if (inlier_mask_E.at<unsigned char>(match_idx))
  {
    double* pt = pointBlockPtr(pt_idx);

    const double w = hpoints4D.at<double>(3, match_idx);
    if (std::fabs(w) > 1e-12)
    {
      pt[0] = hpoints4D.at<double>(0, match_idx) / w;
      pt[1] = hpoints4D.at<double>(1, match_idx) / w;
      pt[2] = hpoints4D.at<double>(2, match_idx) / w;

      if (pt[2] > 0.0)
      {
        cv::Mat_<double> pt_3d =
            (cv::Mat_<double>(3, 1) << pt[0], pt[1], pt[2]);

        pt_3d = init_r_mat * pt_3d + init_t_vec;

        if (pt_3d(2, 0) > 0.0)
        {
          cv::Point2d p0(pt[0] / pt[2], pt[1] / pt[2]);
          cv::Point2d p1(pt_3d(0, 0) / pt_3d(2, 0),
                         pt_3d(1, 0) / pt_3d(2, 0));

          if (cv::norm(p0 - points0[match_idx]) < max_reproj_err_ &&
              cv::norm(p1 - points1[match_idx]) < max_reproj_err_)
          {
            pts_optim_iter_[pt_idx] = 1;
          }
          else
          {
            pts_optim_iter_[pt_idx] = -1;
          }
        }
      }
    }
  }

  match_idx++;
}

  // First bundle adjustment iteration: here we have only two camera poses, i.e., the seed pair
  bundleAdjustmentIter(new_cam_pose_idx );


  // OPTIONAL 2 support data.
  double obs_min_x = std::numeric_limits<double>::max();
  double obs_min_y = std::numeric_limits<double>::max();
  double obs_max_x = -std::numeric_limits<double>::max();
  double obs_max_y = -std::numeric_limits<double>::max();

  for (int i_obs = 0; i_obs < num_observations_; i_obs++)
  {
    const double x = observations_[2 * i_obs];
    const double y = observations_[2 * i_obs + 1];

    obs_min_x = std::min(obs_min_x, x);
    obs_min_y = std::min(obs_min_y, y);
    obs_max_x = std::max(obs_max_x, x);
    obs_max_y = std::max(obs_max_y, y);
  }

  const double obs_range_x = std::max(1e-12, obs_max_x - obs_min_x);
  const double obs_range_y = std::max(1e-12, obs_max_y - obs_min_y);

  // Start to register new poses and observations...
  for(int iter = 1; iter < num_cam_poses_ - 1; iter++ )
  {
    
    // The vector n_init_pts stores the number of points already being optimized
    // that are projected in a new camera pose when is optimized for the first time
    std::vector<int> n_init_pts(num_cam_poses_, 0);
    int max_init_pts = -1;

    const bool use_opt2 = (std::getenv("SFM_opt2") != nullptr &&
                            std::string(std::getenv("SFM_opt2")) == "1");

    // Basic next best view selection strategy.
    if (!use_opt2)
    {
    // Select the new camera (new_cam_pose_idx) to be included in the optimization as the one that has
    // more projected points in common with the cameras already included in the optimization
    for( int i_p = 0; i_p < num_points_; i_p++ )
    {
      if( pts_optim_iter_[i_p] > 0 ) // Point already added
      {
        for(int i_c = 0; i_c < num_cam_poses_; i_c++ )
        {
          if( cam_pose_optim_iter_[i_c] == 0 && // New camera pose not yet registered
              cam_observation_[i_c].find( i_p ) != cam_observation_[i_c].end() ) // Dees camera i_c see this 3D point?
            n_init_pts[i_c]++;
        }
      }
    }

    for(int i_c = 0; i_c < num_cam_poses_; i_c++ )
    {
      if( cam_pose_optim_iter_[i_c] == 0 && n_init_pts[i_c] > max_init_pts )
      {
        max_init_pts = n_init_pts[i_c];
        new_cam_pose_idx = i_c;
      }
    }
    } // end if (!use_opt2)
    
    //////////////////////////// Code to be completed (OPTIONAL 2) ////////////////////////////////
    // Implement an alternative next best view selection strategy, e.g., the one presented
    // in class(see Structure From Motion Revisited paper, sec. 4.2). Just comment the basic next
    // best view selection strategy implemented above and replace it with yours.
    /////////////////////////////////////////////////////////////////////////////////////////
    if (use_opt2)
    {
    std::vector<int> n_init_pts(num_cam_poses_, 0);
    std::vector<double> nbv_score(num_cam_poses_, -1.0);

    double best_nbv_score = -1.0;
    int best_visible_points = -1;
    int best_pnp_inliers = -1;
    new_cam_pose_idx = -1;

    // Pyramid levels for the visibility score
    const int grid_resolutions[3] = {2, 4, 8};

    // candidate selection
    const int min_visible_for_pnp = 20;
    const int min_pnp_inliers = 15;
    const double min_pnp_inlier_ratio = 0.35;

    for (int i_c = 0; i_c < num_cam_poses_; i_c++) {
      if (cam_pose_optim_iter_[i_c] != 0)
      continue;

      std::vector<cv::Point2d> visible_observations;
      std::vector<cv::Point3d> scene_pts_candidate;
      std::vector<cv::Point2d> img_pts_candidate;

      // Collect already reconstructed 3D points seen by this candidate camera
      for (int i_p = 0; i_p < num_points_; i_p++) {
        if (pts_optim_iter_[i_p] > 0 && cam_observation_[i_c].find(i_p) != cam_observation_[i_c].end()) {

          const int obs_idx = cam_observation_[i_c][i_p];

          const double u = observations_[2 * obs_idx];
          const double v = observations_[2 * obs_idx + 1];

          visible_observations.emplace_back(u, v);

          double* pt = pointBlockPtr(i_p);
          scene_pts_candidate.emplace_back(pt[0], pt[1], pt[2]);
          img_pts_candidate.emplace_back(u, v);
        }
      }

      n_init_pts[i_c] = static_cast<int>(visible_observations.size());

      if (n_init_pts[i_c] < min_visible_for_pnp) {
        std::cout << "[NBV_PYRAMID] iter=" << iter
                  << " camera=" << i_c
                  << " visible=" << n_init_pts[i_c]
                  << " score=-1 skipped=too_few_points"
                  << std::endl;
        continue;
      }

      double pyramid_score = 0.0;

      for (int i_res = 0; i_res < 3; i_res++) {

        const int res = grid_resolutions[i_res];
        std::vector<unsigned char> occupied(res * res, 0);

        for (const cv::Point2d& p : visible_observations) {

          int col = static_cast<int>(std::floor((p.x - obs_min_x) / obs_range_x * static_cast<double>(res)));
          int row = static_cast<int>(std::floor((p.y - obs_min_y) / obs_range_y * static_cast<double>(res)));

          col = std::max(0, std::min(res - 1, col));
          row = std::max(0, std::min(res - 1, row));

          occupied[row * res + col] = 1;
        }

        int occupied_cells = 0;
        for (int i_cell = 0; i_cell < res * res; i_cell++) {
          if (occupied[i_cell])
            occupied_cells++;
        }

        const double level_weight = static_cast<double>(res * res);
        pyramid_score += level_weight * static_cast<double>(occupied_cells);
      }

      nbv_score[i_c] = pyramid_score;

      // Validate the candidate with PnP
      cv::Mat candidate_r_vec, candidate_t_vec;
      std::vector<int> pnp_inliers;

      cv::theRNG().state = 12345 + i_c;

      bool pnp_ok = cv::solvePnPRansac(
          scene_pts_candidate,
          img_pts_candidate,
          intrinsics_matrix,
          cv::Mat(),
          candidate_r_vec,
          candidate_t_vec,
          false,
          100,
          max_reproj_err_,
          0.99,
          pnp_inliers
      );

      const int n_pnp_inliers = static_cast<int>(pnp_inliers.size());
      const double pnp_inlier_ratio = static_cast<double>(n_pnp_inliers) / static_cast<double>(std::max(1, n_init_pts[i_c]));

      if (!pnp_ok || n_pnp_inliers < min_pnp_inliers || pnp_inlier_ratio < min_pnp_inlier_ratio) {
        std::cout << "[NBV_PYRAMID] iter=" << iter
                  << " camera=" << i_c
                  << " visible=" << n_init_pts[i_c]
                  << " score=" << nbv_score[i_c]
                  << " pnp_inliers=" << n_pnp_inliers
                  << " ratio=" << pnp_inlier_ratio
                  << " skipped=bad_pnp"
                  << std::endl;
        continue;
      }

      std::cout << "[NBV_PYRAMID] iter=" << iter
                << " camera=" << i_c
                << " visible=" << n_init_pts[i_c]
                << " score=" << nbv_score[i_c]
                << " pnp_inliers=" << n_pnp_inliers
                << " ratio=" << pnp_inlier_ratio
                << std::endl;

      // Main criterion: pyramid visibility score
      // Tie-breakers: more PnP inliers, then more visible points
      if (nbv_score[i_c] > best_nbv_score || (nbv_score[i_c] == best_nbv_score && n_pnp_inliers > best_pnp_inliers) ||
          (nbv_score[i_c] == best_nbv_score && n_pnp_inliers == best_pnp_inliers && n_init_pts[i_c] > best_visible_points)) {
            best_nbv_score = nbv_score[i_c];
            best_visible_points = n_init_pts[i_c];
            best_pnp_inliers = n_pnp_inliers;
            new_cam_pose_idx = i_c;
      }
    }

    if (new_cam_pose_idx < 0) {
      std::cout << "No other positions can be optimized, exiting" << std::endl;
      return false;
    }

    // Reset the RNG so that the PnP call immediately below is reproducible
    // for the camera that was validated here.
    cv::theRNG().state = 12345 + new_cam_pose_idx;

    std::cout << "NBV selected camera " << new_cam_pose_idx
              << " visible=" << n_init_pts[new_cam_pose_idx]
              << " pyramid_score=" << best_nbv_score
              << " pnp_inliers=" << best_pnp_inliers
              << std::endl;
    } // end if (use_opt2)
    /////////////////////////////////////////////////////////////////////////////////////////

    // Now new_cam_pose_idx is the index of the next camera pose to be registered
    // Extract the 3D points that are projected in the new_cam_pose_idx-th pose and that are already registered
    std::vector<cv::Point3d> scene_pts;
    std::vector<cv::Point2d> img_pts;
    for( int i_p = 0; i_p < num_points_; i_p++ )
    {
      if (pts_optim_iter_[i_p] > 0 &&
          cam_observation_[new_cam_pose_idx].find(i_p) != cam_observation_[new_cam_pose_idx].end())
      {
        double *pt = pointBlockPtr(i_p);
        scene_pts.emplace_back(pt[0], pt[1], pt[2]);
        img_pts.emplace_back(observations_[cam_observation_[new_cam_pose_idx][i_p] * 2],
                             observations_[cam_observation_[new_cam_pose_idx][i_p] * 2 + 1]);
      }
    }
    if( scene_pts.size() <= 3 )
    {
      std::cout<<"No other positions can be optimized, exiting"<<std::endl;
      return false;
    }

    // Estimate an initial R,t by using PnP + RANSAC
    cv::solvePnPRansac(scene_pts, img_pts, intrinsics_matrix, cv::Mat(),
                       init_r_vec, init_t_vec, false, 100, max_reproj_err_ );
    // ... and add to the pool of optimized camera positions
    initCamParams(new_cam_pose_idx, init_r_vec, init_t_vec);
    cam_pose_optim_iter_[new_cam_pose_idx] = 1;

    // Extract the new points that, thanks to the new camera, are going to be optimized
    int n_new_pts = 0;
    std::vector<cv::Point2d> points0(1), points1(1);
    cv::Mat_<double> proj_mat0(3, 4), proj_mat1(3, 4), hpoints4D;
    for(int cam_idx = 0; cam_idx < num_cam_poses_; cam_idx++ )
    {
      if( cam_pose_optim_iter_[cam_idx] > 0 )
      {
        for( auto const& co_iter : cam_observation_[cam_idx] )
        {
          auto &pt_idx = co_iter.first;
          if( pts_optim_iter_[pt_idx] == 0 &&
              cam_observation_[new_cam_pose_idx].find(pt_idx ) != cam_observation_[new_cam_pose_idx].end() )
          {
            double *cam0_data = cameraBlockPtr(new_cam_pose_idx),
                *cam1_data = cameraBlockPtr(cam_idx);

            //////////////////////////// Code to be completed (4/7) /////////////////////////////////
            // Triangulate the 3D point with index pt_idx by using the observation of this point
            // in the camera poses with indices new_cam_pose_idx and cam_idx. The pointers
            // cam0_data and cam1_data point to the 6D pose blocks for these inside the parameters
            // vector (e.g., cam0_data[0], cam0_data[1], cam0_data[2] hold the axis-angle
            // representation fo the rotation of the camera with index new_cam_pose_idx.
            // Use the OpenCV cv::triangulatePoints() function, remembering to check the cheirality
            // constraint for both cameras.
            // In case of success (cheirality constrant satisfied) execute the following
            // instructions (decomment e cut&paste):

            // n_new_pts++;
            // pts_optim_iter_[pt_idx] = 1;
            // double *pt = pointBlockPtr(pt_idx);
            // pt[0] = /*X coordinate of the estimated point */;
            // pt[1] = /*X coordinate of the estimated point */;
            // pt[2] = /*X coordinate of the estimated point */;
            /////////////////////////////////////////////////////////////////////////////////////////

            // Do not triangulate a point using the same camera twice.
            if (cam_idx == new_cam_pose_idx)
            continue;

            // Reset projection matrices at each triangulation.
            // Otherwise old values may remain from the previous iteration.
            proj_mat0 = cv::Mat_<double>::zeros(3, 4);
            proj_mat1 = cv::Mat_<double>::zeros(3, 4);

            // Build 3x4 projection matrices [R | t] for both cameras.
            cv::Mat r_vec0 = (cv::Mat_<double>(3,1) <<
            cam0_data[0], cam0_data[1], cam0_data[2]);

            cv::Mat r_vec1 = (cv::Mat_<double>(3,1) <<
            cam1_data[0], cam1_data[1], cam1_data[2]);

            cv::Mat R0, R1;
            cv::Rodrigues(r_vec0, R0);
            cv::Rodrigues(r_vec1, R1);

            cv::Mat t0 = (cv::Mat_<double>(3,1) <<
            cam0_data[3], cam0_data[4], cam0_data[5]);

            cv::Mat t1 = (cv::Mat_<double>(3,1) <<
            cam1_data[3], cam1_data[4], cam1_data[5]);

            R0.copyTo(proj_mat0(cv::Rect(0, 0, 3, 3)));
            t0.copyTo(proj_mat0(cv::Rect(3, 0, 1, 3)));

            R1.copyTo(proj_mat1(cv::Rect(0, 0, 3, 3)));
            t1.copyTo(proj_mat1(cv::Rect(3, 0, 1, 3)));

            // Get the two normalized image observations.
            int obs0 = cam_observation_[new_cam_pose_idx][pt_idx];
            int obs1 = co_iter.second;

            points0[0] = cv::Point2d(
            observations_[2 * obs0],
            observations_[2 * obs0 + 1]);

            points1[0] = cv::Point2d(
            observations_[2 * obs1],
            observations_[2 * obs1 + 1]);

            cv::triangulatePoints(proj_mat0, proj_mat1, points0, points1, hpoints4D);

            // H-normalize the triangulated point.
            double w = hpoints4D.at<double>(3, 0);
            if (std::fabs(w) < 1e-8)
            continue;

            double X = hpoints4D.at<double>(0, 0) / w;
            double Y = hpoints4D.at<double>(1, 0) / w;
            double Z = hpoints4D.at<double>(2, 0) / w;

            cv::Mat_<double> pt3d = (cv::Mat_<double>(3,1) << X, Y, Z);

            // Cheirality check in both cameras.
            cv::Mat_<double> p_in_cam0 = R0 * pt3d + t0;
            cv::Mat_<double> p_in_cam1 = R1 * pt3d + t1;

            if (p_in_cam0(2,0) <= 1e-8 || p_in_cam1(2,0) <= 1e-8)
            continue;

            // Reprojection check in both cameras.
            cv::Point2d reproj0(
            p_in_cam0(0,0) / p_in_cam0(2,0),
            p_in_cam0(1,0) / p_in_cam0(2,0));

            cv::Point2d reproj1(
            p_in_cam1(0,0) / p_in_cam1(2,0),
            p_in_cam1(1,0) / p_in_cam1(2,0));

            double err0 = cv::norm(reproj0 - points0[0]);
            double err1 = cv::norm(reproj1 - points1[0]);

            if (err0 < max_reproj_err_ && err1 < max_reproj_err_)
            {
              n_new_pts++;
              pts_optim_iter_[pt_idx] = 1;

              double *pt = pointBlockPtr(pt_idx);
              pt[0] = X;
              pt[1] = Y;
              pt[2] = Z;
            }

            /////////////////////////////////////////////////////////////////////////////////////////

          }
        }
      }
    }

    cout<<"ADDED "<<n_new_pts<<" new points"<<endl;

    cout << "Using " << iter + 2 << " over " << num_cam_poses_ << " cameras" << endl;
    for(int i = 0; i < int(cam_pose_optim_iter_.size()); i++ )
      cout << int(cam_pose_optim_iter_[i]) << " ";
    cout<<endl;

    // Execute an iteration of bundle adjustment
    bundleAdjustmentIter(new_cam_pose_idx );

    Eigen::Vector3d vol_min = Eigen::Vector3d::Constant((std::numeric_limits<double>::max())),
        vol_max = Eigen::Vector3d::Constant((-std::numeric_limits<double>::max()));
    for(int i_c = 0; i_c < num_cam_poses_; i_c++ )
    {
      if( cam_pose_optim_iter_[i_c] )
      {
        double *camera = cameraBlockPtr (i_c);
        if(camera[3] > vol_max(0)) vol_max(0) = camera[3];
        if(camera[4] > vol_max(1)) vol_max(1) = camera[4];
        if(camera[5] > vol_max(2)) vol_max(2) = camera[5];
        if(camera[3] < vol_min(0)) vol_min(0) = camera[3];
        if(camera[4] < vol_min(1)) vol_min(1) = camera[4];
        if(camera[5] < vol_min(2)) vol_min(2) = camera[5];
      }
    }

    double max_dist = 5*(vol_max - vol_min).norm();
    if(max_dist < 10.0) max_dist = 10.0;

    double *pts = parameters_.data() + num_cam_poses_ * camera_block_size_;
    for( int i = 0; i < num_points_; i++ )
    {
      if( pts_optim_iter_[i] > 0 &&
          ( fabs(pts[i*point_block_size_]) > max_dist ||
              fabs(pts[i*point_block_size_ + 1]) > max_dist ||
              fabs(pts[i*point_block_size_ + 2]) > max_dist ) )
      {
        pts_optim_iter_[i] = -1;
      }
    }
    //////////////////////////// Code to be completed (7/7) //////////////////////////////////
    // The reconstruction may diverge due to outlier triangulations or the bundle adjustment
    // getting stucked in poor local minima. This often manifests as 'exploding' geometry,
    // where camera centers or 3D points are projected to extreme distances from the origin.
    // If such instability is detected, for instance, by monitoring unusually large updates
    // to camera poses or point coordinates during an iteration, the reconstruction should
    // be reset. In this case, the function must return false to trigger a restart with
    // a different seed pair.
    /////////////////////////////////////////////////////////////////////////////////////////
    
    // Controllo 1: Esplosione della scala geometrica
    // Visto che la baseline iniziale è 1.0, se max_dist diventa enorme, 
    // la geometria è matematicamente esplosa (divergenza di Ceres).
    if (max_dist > 100.0) {
        std::cout << "Diverged: geometry exploded (max_dist = " << max_dist << ")" << std::endl;
        return false;
    }

    // Controllo 2: Collasso dei punti
    // Se abbiamo perso quasi tutti i punti a causa degli outlier, scartiamo.
    int n_valid_pts = 0;
    for (int i_p = 0; i_p < num_points_; i_p++) {
        if (pts_optim_iter_[i_p] > 0) n_valid_pts++;
    }
    
    if (n_valid_pts < 10) {
        std::cout << "Diverged: only " << n_valid_pts << " valid points remain." << std::endl;
        return false;
    }

    // Final-iteration metrics (lab spec section 4: comparison tables).
    // The registration loop runs for iter in [1, num_cam_poses_ - 1), so the
    // last body execution is at iter == num_cam_poses_ - 2. We compute the
    // three required quantities (registration success rate, reconstruction
    // density, average reprojection error) only on that final iteration so
    // that they reflect the converged state.
    //
    // The reprojection error is computed with the same projection model as
    // the ReprojectionError functor at the top of this file: rotate the 3D
    // point by camera[0..2] (axis-angle), translate by camera[3..5], then
    // project by dividing x and y by z. The lab spec asks for the MEAN
    // geometric error, which is the average over all observations of the
    // Euclidean distance between observation and projection (not the RMS of
    // the scalar components, which is a different number).
    //
    // The lab spec also asks for the result in pixels, but observations in
    // test_data*.txt are pre-divided by the calibration matrix K (the matcher
    // applies obs_norm = K^-1 * obs_pixel before writing the data file), so
    // residuals come out in normalized canonical coordinates. To recover the
    // pixel error the focal length used during matching must be supplied at
    // runtime through the SFM_FOCAL_PX environment variable. If it is set,
    // the pixel error is printed alongside the normalized error; otherwise
    // only the normalized error is shown together with a reminder.
    if (iter == num_cam_poses_ - 2)
    {
      int n_registered_cams = 0;
      for (int i_c = 0; i_c < num_cam_poses_; i_c++)
      {
        if (cam_pose_optim_iter_[i_c] > 0)
          n_registered_cams++;
      }

      double sum_dist = 0.0;
      long long n_obs = 0;
      for (int i_c = 0; i_c < num_cam_poses_; i_c++)
      {
        if (cam_pose_optim_iter_[i_c] <= 0)
          continue;

        const double *cam = cameraBlockPtr(i_c);

        for (auto const &kv : cam_observation_[i_c])
        {
          const int i_p = kv.first;
          if (pts_optim_iter_[i_p] <= 0)
            continue;

          const double *pt = pointBlockPtr(i_p);

          double rotated[3];
          ceres::AngleAxisRotatePoint(cam, pt, rotated);
          const double pz = rotated[2] + cam[5];
          if (pz <= 0.0 || !std::isfinite(pz))
            continue;
          const double px = (rotated[0] + cam[3]) / pz;
          const double py = (rotated[1] + cam[4]) / pz;

          const int obs_idx = kv.second;
          const double dx = px - observations_[2 * obs_idx];
          const double dy = py - observations_[2 * obs_idx + 1];
          sum_dist += std::sqrt(dx*dx + dy*dy);
          n_obs += 1;
        }
      }

      const double mean_err_norm = (n_obs > 0)
          ? (sum_dist / static_cast<double>(n_obs))
          : 0.0;

      const char *focal_env = std::getenv("SFM_FOCAL_PX");
      const double focal_px = (focal_env != nullptr) ? std::atof(focal_env) : 0.0;

      std::cout << "===== FINAL METRICS =====" << std::endl;
      std::cout << "Registered cameras  : " << n_registered_cams
                << " / " << num_cam_poses_
                << "  (success rate "
                << (100.0 * n_registered_cams / num_cam_poses_) << " %)"
                << std::endl;
      std::cout << "Valid 3D points     : " << n_valid_pts << std::endl;
      if (focal_px > 0.0)
      {
        std::cout << "Mean reprojection err: " << (mean_err_norm * focal_px)
                  << " px  (focal = " << focal_px << " px)" << std::endl;
      }
      else
      {
        std::cout << "Mean reprojection err: " << mean_err_norm
                  << " (normalized image units)" << std::endl;
        std::cout << "  set SFM_FOCAL_PX=<focal length in pixels>"
                  << " before running to print pixel error directly"
                  << std::endl;
      }
      std::cout << "=========================" << std::endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////
  }

  return true;

}

void BasicSfM::initCamParams(int new_pose_idx, cv::Mat r_vec, cv::Mat t_vec )
{
  double *camera = cameraBlockPtr(new_pose_idx);

  cv::Mat_<double> r_vec_d(r_vec), t_vec_d(t_vec);
  for( int r = 0; r < 3; r++ )
  {
    camera[r] = r_vec_d(r,0);
    camera[r+3] = t_vec_d(r,0);
  }
}

void BasicSfM::bundleAdjustmentIter( int new_cam_idx )
{
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 200;

  std::vector<double> bck_parameters;

  bool keep_optimize = true;

  // Global optimization
  while( keep_optimize )
  {
    bck_parameters = parameters_;
    ceres::Problem problem;
    ceres::Solver::Summary summary;

    // For each observation....
    for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
    {
      //.. check if this observation has bem already registered (both checking camera pose and point pose)
      if( cam_pose_optim_iter_[cam_pose_index_[i_obs]] > 0 && pts_optim_iter_[point_index_[i_obs]] > 0 )
      {
        //////////////////////////// Code to be completed (6/7) /////////////////////////////////
        //... in case, add a residual block inside the Ceres solver problem.
        // You should define a suitable functor (i.e., use the ReprojectionError struct at the
        // beginning of this file)
        // Remember that the parameter blocks are stored starting from the
        // parameters_.data() double* pointer.
        // The camera position blocks have size (camera_block_size_) of 6 elements,
        // while the point position blocks have size (point_block_size_) of 3 elements.
        //
        // To handle outlier matches, you are required to experiment with different robust loss
        // kernels. Compare the standard 'NULL' loss (least squares) with robust alternatives
        // such as:
        // - ceres::HuberLoss(a)
        // - ceres::CauchyLoss(a)
        // where 'a' is a scale parameter (e.g., 1.0 or 2 * max_reproj_err_).
        //////////////////////////////////////////////////////////////////////////////////

        // Pointer to the 2D observation (canonical (x, y) pair).
        double *observation = observations_.data() + (i_obs * 2);

        // Auto-diff cost function for this observation.
        ceres::CostFunction *cost_function =
            ReprojectionError::Create(observation[0], observation[1]);

        // Robust loss kernel (env-tunable for experimentation).
        // SFM_LOSS:  one of {cauchy, huber, null}. Default: cauchy.
        //   - "null"   -> nullptr, plain least-squares (no kernel).
        //   - "huber"  -> ceres::HuberLoss(SFM_SCALE * max_reproj_err_).
        //   - "cauchy" -> ceres::CauchyLoss(SFM_SCALE * max_reproj_err_).
        // SFM_SCALE: scale factor multiplied by max_reproj_err_. Default: 2.0,
        //            as suggested by the lab spec hint.
        ceres::LossFunction *loss_function;
        {
          const char *_envloss  = std::getenv("SFM_LOSS");
          const char *_envscale = std::getenv("SFM_SCALE");
          std::string _lk = _envloss  ? std::string(_envloss) : std::string("cauchy");
          double      _sc = _envscale ? std::atof(_envscale)  : 2.0;
          if      (_lk == "null")  loss_function = nullptr;
          else if (_lk == "huber") loss_function = new ceres::HuberLoss (_sc * max_reproj_err_);
          else                     loss_function = new ceres::CauchyLoss(_sc * max_reproj_err_);
        }

        // Pointers into the contiguous parameter vector.
        double *camera_block = parameters_.data()
                               + camera_block_size_ * cam_pose_index_[i_obs];
        double *point_block  = parameters_.data()
                               + camera_block_size_ * num_cam_poses_
                               + point_block_size_  * point_index_[i_obs];

        problem.AddResidualBlock(cost_function, loss_function,
                                 camera_block, point_block);

        /////////////////////////////////////////////////////////////////////////////////////////

      }
    }

    Solve(options, &problem, &summary);

    // WARNING Here poor optimization ... :(
    // Check the cheirality constraint
    int n_cheirality_violation = 0;
    for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
    {
      if( cam_pose_optim_iter_[cam_pose_index_[i_obs]] > 0 &&
          pts_optim_iter_[point_index_[i_obs]] == 1 &&
          !checkCheiralityConstraint(cam_pose_index_[i_obs], point_index_[i_obs]))
      {
        // Penalize the point..
        pts_optim_iter_[point_index_[i_obs]] -= 2;
        n_cheirality_violation++;
      }
    }

    int n_outliers;
    if( n_cheirality_violation > max_outliers_ )
    {
      std::cout << "****************** OPTIM CHEIRALITY VIOLATION for " << n_cheirality_violation << " points : redoing optim!!" << std::endl;
      parameters_ = bck_parameters;
    }
    else if ( (n_outliers = rejectOuliers()) > max_outliers_ )
    {
      std::cout<<"****************** OPTIM FOUND "<<n_outliers<<" OUTLIERS : redoing optim!!"<<std::endl;
      parameters_ = bck_parameters;
    }
    else
      keep_optimize = false;
  }

  printPose ( new_cam_idx );
}

int BasicSfM:: rejectOuliers()
{
  int num_ouliers = 0;
  for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
  {
    if( cam_pose_optim_iter_[cam_pose_index_[i_obs]] > 0 && pts_optim_iter_[point_index_[i_obs]] > 0 )
    {
      double *camera = cameraBlockPtr (cam_pose_index_[i_obs]),
             *point = pointBlockPtr (point_index_[i_obs]),
             *observation = observations_.data() + (i_obs * 2);

      double p[3];
      ceres::AngleAxisRotatePoint(camera, point, p);

      // camera[3,4,5] are the translation.
      p[0] += camera[3];
      p[1] += camera[4];
      p[2] += camera[5];

      double predicted_x = p[0] / p[2];
      double predicted_y = p[1] / p[2];

      if ( fabs(predicted_x - observation[0]) > max_reproj_err_ ||
           fabs(predicted_y - observation[1]) > max_reproj_err_ )
      {
        // Penalize the point
        pts_optim_iter_[point_index_[i_obs]]-=2;
        num_ouliers ++;
      }
    }
  }
  return num_ouliers;
}
