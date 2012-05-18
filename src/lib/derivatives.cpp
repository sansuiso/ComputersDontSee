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

#include "derivatives.hpp"

void cds::HorizontalGradientWithBackwardScheme(cv::Mat const &X, cv::Mat &Dx)
{
    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));
    
    int valuesPerRow = (X.cols-1) * X.channels();
    
    for (int i=0; i<X.rows; ++i) 
    {
        const float *xjm1 = X.ptr<float>(i);
        const float *xj = xjm1 + X.channels();
        
        float *pdx = Dx.ptr<float>(i) + X.channels();
        
        for (int j = 0; j < valuesPerRow; ++j, ++xj, ++xjm1, ++pdx)
            *pdx = (*xj - *xjm1);
    }
}

void cds::VerticalGradientWithBackwardScheme(cv::Mat const &X, cv::Mat &Dx)
{
    if (!X.data)
        return;

    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));
    
    int valuesPerRow = X.channels() * X.cols;
    
    for (int i = 1; i < X.rows; ++i) 
    {
        // Current row
        const float *xi = X.ptr<float>(i);
        // Previous row
        const float *xim1 = X.ptr<float>(i-1);
        
        float *pdy = Dx.ptr<float>(i);
        
        for (int j = 0; j < valuesPerRow; ++j, ++xi, ++xim1, ++pdy)
            *pdy = (*xi - *xim1);
    }
}


void cds::DivergenceWithBackwardScheme(cv::Mat const &X1, cv::Mat const &X2, cv::Mat &divX)
{	
    if (!X1.data || !X2.data)
    {
        return;
    }
    
	divX.create(X1.size(), CV_32FC1);
	divX.setTo(cv::Scalar::all(0));
    
	cv::Mat DX1;
    cds::HorizontalGradientWithBackwardScheme(X1, DX1);
	
	cv::Mat DX2;
    cds::VerticalGradientWithBackwardScheme(X2, DX2);
	
	divX = DX1 + DX2;
}

void cds::HorizontalGradientWithForwardScheme(cv::Mat const &X, cv::Mat &Dx)
{
    if (!X.data)
    {
        return;
    }
    
    Dx.create(X.size(), CV_32FC1);
    Dx.setTo(cv::Scalar::all(0));
    
    for (int i=0; i<X.rows; ++i) 
    {
        register const float *xj = X.ptr<float>(i);
        register const float *xjp1 = xj + X.channels();
        
        register float *pdx = Dx.ptr<float>(i);
        
        for (int j=0; j < X.cols; ++j, ++xj, ++xjp1, ++pdx)
            *pdx = (*xjp1 - *xj);
    }
}

void cds::VerticalGradientWithForwardScheme(cv::Mat const &X, cv::Mat &Dx)
{
    if (!X.data)
        return;
    
    Dx.create(X.size(), CV_32FC1);
    Dx.setTo(cv::Scalar::all(0));
    
    int valuesPerRow = X.cols * X.channels();
    
    for (int i=0; i<X.rows-1; ++i) 
    {
        // Current row
        const float *xi = X.ptr<float>(i);
        // Next row
        const float *xip1 = X.ptr<float>(i+1);
        
        float *pdy = Dx.ptr<float>(i);
        
        for (int j = 0; j < X.cols; ++j, ++xi, ++xip1, ++pdy)
            *pdy = (*xip1 - *xi);
    }
}