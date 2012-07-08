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

#ifndef CDS_RIPPLES_HPP
#define CDS_RIPPLES_HPP

#include <opencv2/core/core.hpp>

namespace cds
{
  namespace ripples
  {
    int haar(cv::Mat const &anImage, cv::Mat &analysis, int maxLevels, int direction=0);
    int ihaar(cv::Mat const &coefficients, cv::Mat &synthesis, int maxLevels);

    int daubechies4(cv::Mat const &anImage, cv::Mat &analysis, int maxLevels, int direction=0);
    int idaubechies4(cv::Mat const &coefficients, cv::Mat &synthesis, int maxLevels);

    int cdf46(cv::Mat const &anImage, cv::Mat &analysis, int maxLevels, int direction=0);
    int icdf46(cv::Mat const &coefficients, cv::Mat &synthesis, int maxLevels);
  }
}

#endif	// CDS_FOURIER_HPP
