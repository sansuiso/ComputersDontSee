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

#ifndef CDS_PROX_HPP
#define CDS_PROX_HPP

#include <opencv2/core/core.hpp>

namespace cds
{
	/**
	 *
	 */
	void ProxL2Ball(cv::Mat &X, cv::Mat const &center=cv::Mat(), float radius=1.0);

	/**
	 * Solves the proximal operator associated with the L2 dataterm, i.e. solves the problem:
	 * 		argmin { |Y-X|^2/(2*tau) + lambda*|Y-dataTerm|^2 }
	 * X is modified by the function, i.e. contains the solution.
	 */
	void ProxL2(cv::Mat &X, cv::Mat const &dataTerm, float lambda, float tau);


	/**
	 * Solves the proximal operator associated with the L2 inpainting dataterm, i.e. solves the problem:
	 * 		argmin { |Y-X|^2/(2*tau) + lambda*|A(Y)-dataTerm|^2 }
	 * X is modified by the function, i.e. contains the solution.
	 */
	void ProxL2Inpainting(cv::Mat &X, cv::Mat const &dataTerm, cv::Mat const &mask);
	
	/**
	 *
	 */
	void ProxLinfBall(cv::Mat &X, cv::Mat const &center=cv::Mat(), float radius=1.0);

	/**
	 *
	 */
	void ProxLinfBall(cv::Mat &X1, cv::Mat &X2, cv::Mat const &C1=cv::Mat(), cv::Mat const &C2=cv::Mat(), float radius=1.0);

	/**
	 * Computes the proximal operator of the indicator function of the set [xmin,xmax],
	 * i.e. thresholds x to be in this set.
	 */
	void ProxInterval(cv::Mat &X, float xmin=0.0, float xmax=1.0);
}

#endif	// CDS_PROX_HPP