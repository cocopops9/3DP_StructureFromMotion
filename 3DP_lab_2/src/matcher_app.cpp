#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "io_utils.h"

#include "features_matcher.h"

int main(int argc, char **argv)
{
  if( argc < 5 )
  {
    std::cout<<"Usage : "<<argv[0]<<" <calibration parameters filename> <images folder filename>"
                          <<"<output data file> <use modern features flag (0 or 1)> [<focal length scale>]"<<std::endl;
    return 0;
  }
  std::string results_file(argv[3]);

  bool use_modern_features = atof(argv[4]);
  double focal_scale = 1.0;
  if( argc > 5 )
    focal_scale = atof(argv[5]);

  cv::Size image_size;
  cv::Mat intrinsics_matrix, dist_coeffs;

  if( !loadCameraParams( argv[1], image_size, intrinsics_matrix, dist_coeffs ) )
  {
    std::cerr<<"Can't load calibration parameters, exiting"<<std::endl;
    return -1;
  }

  std::cout<<"Image size : "<<image_size<<std::endl;
  std::cout<<"intrinsics matrix :"<<std::endl<<intrinsics_matrix<<std::endl;
  std::cout<<"Distortion coefficients :"<<std::endl<<dist_coeffs<<std::endl;

  if(use_modern_features)
    std::cout<<"Using classical features"<<std::endl;
  else
    std::cout<<"Using modern features"<<std::endl;

  std::cout<<"Focal length scale :"<<std::endl<<focal_scale<<std::endl;

  std::vector<std::string> images_names;
  if( !readFileNamesFromFolder ( argv[2], images_names ) )
  {
    std::cerr<<"Can't load images names, exiting"<<std::endl;
    return -1;
  }

  FeatureMatcher matcher(intrinsics_matrix, dist_coeffs, use_modern_features, focal_scale );
  matcher.setImagesNames(images_names);
  matcher.extractFeatures();
  matcher.exhaustiveMatching();
  std::cout<<"Exhaustive matching done!"<<std::endl;
  matcher.writeToFile(results_file, true);
  std::cout<<"Results saved to "<<results_file<<std::endl;
  std::cout<<"Type any key to check matches, ESC to exit"<<std::endl;

  matcher.testMatches();

  return 0;
}
