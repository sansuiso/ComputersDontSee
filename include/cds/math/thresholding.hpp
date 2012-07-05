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

#ifndef CDS_SOFTTHRESHOLDING_HPP
#define CDS_SOFTTHRESHOLDING_HPP

#include <opencv2/core/core.hpp>

namespace cds
{
  /**
   * @brief Soft-thresholding of a vector
   *
   * Computes the soft-thresholding of a vector X w.r.t. to a given treshold.
   * For now, it calls the more general function with a varying per-pixel threshold, but 
   * in the future more specific (optimized) versions may be used here.
   *
   * @param X Values to threshold. Should be of type CV_32FC1.
   * @param threshold The threshold of the soft-thresholding.
   *
   * @see softThresholding(cv::Mat &X, cv::Mat const &thresholds)
   */
  void softThresholding(cv::Mat &X, float threshold);

  /**
   * @brief General function of soft-thresholding
   *
   * Computes the soft thresholding of a vector X.<br />
   * The value is computed by shrinkage, i.e. S(x) = x*(1 - threshold/|x|)^+,
   * instead if the usual S(x) = sign(x)*(|x| - threshold)^+. This may change in the future
   * if good reasons for this arise.
   * The threshold can vary on a per-pixel basis.
   * All the matrices should be of type CV_32FC1.
   *
   * @param X The matrix to soft-threshold
   * @param thresholds The matrix of the thresholds.
   */
  void softThresholding(cv::Mat &X, cv::Mat const &thresholds);

  /**
   * Hard thresholding of a matrix of coefficients.
   *
   * Hard thresholding means that coefficients whose module is smaller than
   * the given threshold are set to 0, otherwise they are left untouched.
   * If the matrix has several channels, only the first one is processed.
   *
   * @param X Matrix of coefficients of type CV_32F
   * @param threshold The hard thresholding threshold
   */
  void hardThresholding(cv::Mat &X, float threshold);
}

#endif	// CDS_SOFTTHRESHOLDING_HPP
