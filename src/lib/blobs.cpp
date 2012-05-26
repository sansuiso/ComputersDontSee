#include "blobs.hpp"
#include <opencv2/imgproc/imgproc.hpp>

void cds::Binarize(cv::Mat const &anImage, cv::Mat &binaryResult)
{
  binaryResult.create(anImage.size(), anImage.type());
  cv::threshold(anImage, binaryResult, 128, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);
}

void cds::CleanDetectionMask(cv::Mat &foreground)
{
  // First, get rid of peaks
  cv::medianBlur(foreground, foreground, 5);

  // Now, apply erosion then dilation
  for (int i = 0; i < 2; ++i)
  {
    cv::erode(foreground, foreground, cv::Mat());
  }

  for (int i = 0; i < 2; ++i)
  {
    cv::dilate(foreground, foreground, cv::Mat());
  }
}

void cds::MarkDetection(cv::Mat const &foreground, cv::Mat &drawImage)
{
  int minX, minY, maxX, maxY;
  minX = foreground.cols;
  maxX = 0;
  minY = foreground.rows;
  maxY = 0;
  
  // Get the bounding box
  for (int y = 0; y < foreground.rows; ++y)
  {
    unsigned char const *p_fg = foreground.ptr<uchar>(y);
    
    for (int x = 0; x < foreground.cols; ++x)
    {
      if (*p_fg++ > 0)
      {
	minX = MIN(x, minX);
	maxX = MAX(x, maxX);
	minY = MIN(y, minY);
	maxY = MAX(y, maxY);
      }
    }
  }

  // Draw a red rectangle
  cv::rectangle(drawImage, cv::Point(minX, minY), cv::Point(maxX, maxY), CV_RGB(255,0,0), 2, CV_AA);
}
