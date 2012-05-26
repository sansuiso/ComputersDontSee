#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include "../lib/include/blobs.hpp"

#define LEARNINGRATE_TRACK_LENGTH 51

int iLearningRate = 1;
double learningRate = 1.0;

void on_trackRate(int, void *)
{
  learningRate = double(iLearningRate)/(double)LEARNINGRATE_TRACK_LENGTH;
}

int main(int argc, char *argv[])
{
  // Read input video from the command line
  cv::VideoCapture inputMovie;
  bool success = inputMovie.open(argv[argc-1]);
  if (!success)
  {
    std::cerr << "Failed to open: " << argv[argc-1] << std::endl;
    return EXIT_FAILURE;
  }

  // Input data
  cv::namedWindow("Original", CV_WINDOW_KEEPRATIO);
  cv::Mat currentFrame;

  // BG subtractor
  cv::BackgroundSubtractorMOG2 bgSuber;

  // Result
  cv::Mat foreground;
  cv::Mat background;
  cv::namedWindow("Foreground", CV_WINDOW_KEEPRATIO);
  cv::namedWindow("Background", CV_WINDOW_KEEPRATIO);
  cv::namedWindow("Result", CV_WINDOW_KEEPRATIO);

  // GUI
  cv::createTrackbar("Learning rate", "Background", &iLearningRate, LEARNINGRATE_TRACK_LENGTH, on_trackRate);

  // Main loop
  inputMovie >> currentFrame;
  while(currentFrame.data)
  {
    // Display input frame
    cv::imshow("Original", currentFrame);
    
    // Detect
    bgSuber(currentFrame, foreground, learningRate-1);
    bgSuber.getBackgroundImage(background);

    // Binarize FG
    cds::Binarize(foreground, foreground);
    cds::CleanDetectionMask(foreground);

    // Get main object
    cds::MarkDetection(foreground, currentFrame);
    cv::imshow("Result", currentFrame);

    // Showtime
    cv::imshow("Foreground", foreground);
    cv::imshow("Background", background);

    // Next frame
    int key = cv::waitKey();
    if (key != 27)
    {
      inputMovie >> currentFrame;
    }
    else
    {
      currentFrame.release();
    }
  }

  // Done, exit
  return EXIT_SUCCESS;
}
