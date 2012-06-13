// Copyright (c) 2012 D'ANGELO Emmanuel                                                                                                    
// All rights reserved.                                                                                                                    
//                                                                                                                                         
// Redistribution and use in source and binary forms, with or without modification,                                                        
// are permitted provided that the following conditions are met:                                                                           

// * Redistributions of source code must retain the above copyright notice, this list of conditions                                        
//   and the following disclaimer.                                                                                                         
// * Redistributions in binary form must reproduce the above copyright notice, this list of conditions                                     
//   and the following disclaimer in the documentation and/or other materials provided with the distribution.                              
// * Neither the name of the copyright holder nor the names of its contributors may be used                                                
//   to endorse or promote products derived from this software without specific prior written permission.                                  
//                                                                                                                                         
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"                                                             
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,                                                                   
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.                                          
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                                               
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,                                                
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)                                   
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,                                                   
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,                                         
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

#include <cds/motion/blobs.hpp>
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
