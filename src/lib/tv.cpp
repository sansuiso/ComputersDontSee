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

#include "tv.hpp"
#include "derivatives.hpp"
#include "prox.hpp"

#include <iostream>

void cds::TvDiffusion(cv::Mat const &g, cv::Mat &u, int iterations, float lambda)
{
	if(!g.data)
	{
		return;
	}
	
	if (!u.data)
	{
		u = cv::Mat::zeros(g.size(), CV_32FC1);
	}
	
    u.create(g.size(), CV_32FC1);
    
    // Numerical parameters
    float L2 = 8.0f;
    float tau = 1.0f / std::sqrt(L2);
    float sigma = 1.0f / std::sqrt(L2);
    float lambdaTau = lambda * tau;
    
    // Auxiliary points
    cv::Mat ubar, u_nm1;
    cv::Mat Du1, Du2;
    
    // Dual variable
    cv::Mat p1, p2;
    p1.create(g.size(), CV_32FC1);
    p2.create(g.size(), CV_32FC1);
    
    u.copyTo(ubar);
    u.copyTo(u_nm1);
    
    cds::HorizontalGradientWithForwardScheme(u, p1);
    cds::VerticalGradientWithForwardScheme(u, p2);
    
    for (int iter = 0; iter < iterations; ++iter)
    {
        // Update the dual variable
        cds::HorizontalGradientWithForwardScheme(ubar, Du1);
        p1 += sigma * Du1;
        
        cds::VerticalGradientWithForwardScheme(ubar, Du2);
        p2 += sigma * Du2;
        
		cds::ProxLinfBall(p1, p2);

        // Update the solution
        cds::DivergenceWithBackwardScheme(p1, p2, u);
        u *= tau;
        u += u_nm1;
        cds::ProxL2(u, g, lambda, tau);
        
        // Update the auxiliary point
        ubar = 2.0f * u - u_nm1;
        u.copyTo(u_nm1);
    }
}

void cds::TvInpainting(cv::Mat const &g, cv::Mat const &mask, cv::Mat &u, int iterations)
{
	if(!g.data || !mask.data)
	{
		return;
	}
	
	if (!u.data)
	{
		u = cv::Mat::zeros(g.size(), CV_32FC1);
	}
	
    u.create(g.size(), CV_32FC1);
    
    cv::Mat invMask;
    cv::absdiff(mask, cv::Scalar::all(-1), invMask);
    
    // Numerical parameters
    float L2 = 8.0f;
    float tau = 1.0f / std::sqrt(L2);
    float sigma = 1.0f / std::sqrt(L2);
    
    // Auxiliary points
    cv::Mat ubar, u_nm1;
    cv::Mat Du1, Du2;
    
    // Dual variable
    cv::Mat p1, p2;
    p1.create(g.size(), CV_32FC1);
    p2.create(g.size(), CV_32FC1);
    
    u.copyTo(ubar);
    u.copyTo(u_nm1);
    
    cds::HorizontalGradientWithForwardScheme(u, p1);
    cds::VerticalGradientWithForwardScheme(u, p2);
    
    for (int iter = 0; iter < iterations; ++iter)
    {
        // Update the dual variable
        cds::HorizontalGradientWithForwardScheme(ubar, Du1);
        p1 += sigma * Du1;
        
        cds::VerticalGradientWithForwardScheme(ubar, Du2);
        p2 += sigma * Du2;
        
		cds::ProxLinfBall(p1, p2);
        
        // Update the solution
        cds::DivergenceWithBackwardScheme(p1, p2, u);
        u *= tau;
        u += u_nm1;
        cds::ProxL2Inpainting(u, g, mask);
        cds::ProxInterval(u, 0.0, 1.0);
        
        // Update the auxiliary point
        ubar = 2.0f * u - u_nm1;
        u.copyTo(u_nm1);
    }
}
