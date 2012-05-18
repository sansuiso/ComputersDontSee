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

#ifndef CDS_DERIVATIVES_HPP
#define CDS_DERIVATIVES_HPP

#include <opencv2/core/core.hpp>

namespace cds
{
	/**
	 * Divergence of a vector field defines by its 2 components X1 and X2, using a backward scheme
	 * @param X1 Single-channel, floating-point image of the first component
	 * @param X2 Single-channel, floating-point image of the first component
	 * @param divX Single-channel, floating-point image of the divergence of X=(X1,X2)
	 */
	void DivergenceWithBackwardScheme(cv::Mat const &X1, cv::Mat const &X2, cv::Mat &divX);
	
	/**
	 * Horizontal gradient of a single-channel image X using a forward scheme
	 * @param[in] X Single-channel, floating-point image
	 * @param[out] Dx Horizontal component of the gradient of X
	 * @see HorizontalGradientWithBackwardScheme
	 * @see HorizontalGradientWithBackwardScheme
	 */
	void HorizontalGradientWithForwardScheme(cv::Mat const &X, cv::Mat &Dx);
	
	/**
	 * Vertical gradient of a single-channel image X using a forward scheme
	 * @param[in] X Single-channel, floating-point image
	 * @param[out] Dx Vertical component of the gradient of X
	 * @see VerticalGradientWithBackwardScheme
	 */
	void VerticalGradientWithForwardScheme(cv::Mat const &X, cv::Mat &Dx);
	
	/**
	 * Horizontal gradient of a single-channel image X using a backward scheme
	 * @param[in] X Single-channel, floating-point image
	 * @param[out] Dx Horizontal component of the gradient of X
	 * @see HorizontalGradientWithForwardScheme
	 */
	void HorizontalGradientWithBackwardScheme(cv::Mat const &X, cv::Mat &Dx);
	
	/**
	 * Vertical gradient of a single-channel image X using a backward scheme
	 * @param[in] X Single-channel, floating-point image
	 * @param[out] Dx Vertical component of the gradient of X
	 * @see VerticalGradientWithForwardScheme
	 */
	void VerticalGradientWithBackwardScheme(cv::Mat const &X, cv::Mat &Dx);
}
	
#endif	// CDS_DERIVATIVES_HPP