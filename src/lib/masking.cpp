// Copyright (c) 2012 D'ANGELO Emmanuel
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
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

#include "masking.hpp"

void cds::CreateRandomMask(cv::Size frameSize, float occlusionRatio, cv::Mat &mask)
{
  mask.create(frameSize, CV_32FC1);
  mask.setTo(cv::Scalar(1));

  occlusionRatio = MAX(MIN(occlusionRatio, 1.0), 0.0);
  
  // We generate a mask of uniform random values in [0,1] then binarize w.r.t. the occlusionRatio
  cv::randu(mask, cv::Scalar(0), cv::Scalar(1));

  // This should imply a faster loop if applicable
  if (mask.isContinuous()) 
  {
    frameSize.width = frameSize.area();
    frameSize.height = 1;
  }

  // Loop over the mask pixels
  for (int y = 0; y < frameSize.height; ++y) 
  {  
    // Get a pointer to the first pixel of the current row
    float *p_mask = mask.ptr<float>(y);
    
    for (int x = 0; x < frameSize.width; ++x) 
    {
      if (*p_mask < occlusionRatio) 
      {
		*p_mask++ = 0.0;
      } 
      else 
      {
		*p_mask++ = 1.0;
      }
    } // End loop over the colums
  } // End loop over the rows
  
  // Done
}

void cds::CreateInterleavedRowsMask(cv::Size frameSize, cv::Mat &mask)
{
  mask.create(frameSize, CV_32FC1);
  mask.setTo(cv::Scalar(1));
  
  // Mask every odd row
  for (int row = 1; row < frameSize.height; row += 2) 
  {
    cv::Mat currentRow = mask.rowRange(row, row+1);
    currentRow.setTo(cv::Scalar(0));
  }
}

void cds::CreateRectangularMask(cv::Size frameSize, cv::Rect const &rectangle, cv::Mat &mask)
{
  mask.create(frameSize, CV_32FC1);
  mask.setTo(cv::Scalar(1));
  
  // Adapt the mask to the frame size
  cv::Rect maskingRect = rectangle;
  
  if (maskingRect.x < 0 || maskingRect.x >= frameSize.width) 
  {
    return;
  }
  
  // Mask the desired rectangle
  cv::Mat ROI = mask(rectangle);
  ROI.setTo(cv::Scalar(0));
}

