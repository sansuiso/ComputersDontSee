#ifndef CDS_BLOBS_HPP
#define CDS_BLOBS_HPP

#include <opencv2/core/core.hpp>

namespace cds
{
  void Binarize(cv::Mat const &anImage, cv::Mat &binaryResult);
  void CleanDetectionMask(cv::Mat &foreground);
  void MarkDetection(cv::Mat const &foreground, cv::Mat &drawImage);
}

#endif  // CDS_BLOBS_HPP
