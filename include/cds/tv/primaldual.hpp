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

#ifndef CDS_PRIMALDUAL_HPP
#define CDS_PRIMALDUAL_HPP

#include <opencv2/core/core.hpp>

namespace cds
{
  /**
   * Solves the Rudin-Osher-Fatemi denoising problem: 
   * 		min 0.5*lambda*|u-g|^2 + TV(u) 
   * using the primal-dual scheme in [1].
   * TV is the isotropic Total Variation sqrt(|ux|^2 + |uy|^2), where ux and uy are the derivatives
   * of u with respect to x and y respectively.
   *
   * @param g The observed image of type CV_32FC1
   * @param u The resulting image of type CV_32FC1
   * @param iterations The number of iterations of the algorithm (25-100 are good values)
   * @param lambda Weight of the data term
   */
  void TvDiffusion(cv::Mat const &g, cv::Mat &u, int iterations, float lambda);
  
  /**
   * Solves the TV-L2 inpainting problem: 
   * 		min 0.5*lambda*|Au-g|^2 + TV(u) 
   * using the primal-dual scheme in [1].
   * TV is the isotropic Total Variation sqrt(|ux|^2 + |uy|^2), where ux and uy are the derivatives
   * of u with respect to x and y respectively.
   * A is a masking operator.<br />
   * The scheme assumes that lambda is 0 when the mask is 0, and infinite when the mask is 1, i.e.
   * that the result matches perfectly the observation when the mask is not opaque and is free to evolve
   * otherwise.
   *
   * @param g The observed image of type CV_32FC1
   * @param mask The mask image, values in {0,1}, of type CV_32FC1
   * @param u The resulting image of type CV_32FC1
   * @param iterations The number of iterations of the algorithm (25-100 are good values)
   */
  void TvInpainting(cv::Mat const &g, cv::Mat const &mask, cv::Mat &u, int iterations);		
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// REFERENCES:																					//
//																								//
// [1] Chambolle, A., Pock, T. (2010).	 														//
//     A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging. 	//
//     Journal of Mathematical Imaging and Vision, 40(1), 120–145.								//
//////////////////////////////////////////////////////////////////////////////////////////////////

#endif	// CDS_PRIMALDUAL_HPP
