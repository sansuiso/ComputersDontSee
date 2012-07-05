#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <cds/math/thresholding.hpp>

#ifndef CDS_EPS
#define CDS_EPS 1e-6
#endif

void cds::softThresholding(cv::Mat &X, float threshold)
{
  if (!X.data || threshold < 0.0)
  {
    return;
  }

  cv::Mat thresholds(X.size(), X.type(), cv::Scalar::all(threshold));  
  cds::softThresholding(X, thresholds);
}

void cds::softThresholding(cv::Mat &X, cv::Mat const &thresholds)
{
  CV_Assert(X.data != 0 && X.type() == CV_32FC1);
  CV_Assert(X.size() == thresholds.size() && X.type() == thresholds.type());

  for (int y = 0; y < X.rows; ++y)
  {
    float *p_x = X.ptr<float>(y);
    float const *p_thresh = thresholds.ptr<float>(y);

    for (int x = 0; x < X.cols; ++x)
    {
      float absX = fabs(*p_x);
      float shrinkage = 0.0;

      if (absX > CDS_EPS)
      {
	shrinkage = MAX(0.0, 1.0 - (*p_thresh)/absX);
      }

      *p_x++ *= shrinkage;
      ++p_thresh;
    }
  }
}

void cds::hardThresholding(cv::Mat &X, float threshold)
{
  cv::Size workingSize = X.size();
  if (X.isContinuous())
  {
    workingSize.width *= workingSize.height;
    workingSize.height = 1;
  }
  
  for (int y = 0; y < workingSize.height; ++y)
  {
    float *p_x = X.ptr<float>(y);

    for (int x = 0; x < workingSize.width; ++x)
    {
      float absX = fabs(*p_x);
      *p_x++ *= (absX > threshold);
    }
  }
}
