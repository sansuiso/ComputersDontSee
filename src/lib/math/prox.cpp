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

#include <cds/math/prox.hpp>

#define SQUARED_NORM(X,Y) ((X)*(X)+(Y)*(Y))

namespace cds
{
	void ProxL2UnitBall(cv::Mat &X);
	void ProxLinfUnitBall(cv::Mat &X);
	void ProxLinfUnitBall(cv::Mat &X1, cv::Mat &X2);
}

void cds::ProxL2Ball(cv::Mat &X, cv::Mat const &center, float radius)
{
	if (!X.data)
	{
		return;
	}
	
	// Shift the center
	if (center.data)
	{
		X -= center;
	}
	
	// Normalize the coordinates
	X /= radius;
	
	// Apply projection
	cds::ProxL2UnitBall(X);

	// Un-normalize
	X *= radius;
	
	// Shift back
	if (center.data)
	{
		X += center;
	}	
}

void cds::ProxL2(cv::Mat &X, cv::Mat const &dataTerm, float lambda, float tau)
{
	float lambdaTau = lambda*tau;
	
    X += (lambdaTau * dataTerm);
    X /= (1.0 + lambdaTau);
}


void cds::ProxLinfBall(cv::Mat &X, cv::Mat const &center, float radius)
{
	if (!X.data)
	{
		return;
	}
	
	// Shift the center
	if (center.data)
	{
		X -= center;
	}
	
	// Normalize the coordinates
	X /= radius;
	
	// Apply projection
	cds::ProxLinfUnitBall(X);

	// Un-normalize
	X *= radius;
	
	// Shift back
	if (center.data)
	{
		X += center;
	}
}

void cds::ProxLinfBall(cv::Mat &X1, cv::Mat &X2, cv::Mat const &C1, cv::Mat const &C2, float radius)
{
	if (!X1.data)
	{
		return;
	}
	
	// Shift the center
	if (C1.data)
	{
		X1 -= C1;
	}
	
	if (C2.data)
	{
		X2 -= C2;
	}
	
	// Normalize the coordinates
	X1 /= radius;
	X2 /= radius;
	
	// Apply projection
	cds::ProxLinfUnitBall(X1, X2);

	// Un-normalize
	X1 *= radius;
	X2 *= radius;
	
	// Shift back
	if (C1.data)
	{
		X1 += C1;
	}
	if (C2.data)
	{
		X2 += C2;
	}
}

void cds::ProxL2UnitBall(cv::Mat &X)
{
	if (!X.data)
	{
		return;
	}
	
	for (int y = 0; y < X.rows; ++y)
	{
		float *p_x = X.ptr<float>(y);
		
		for (int x = 0; x < X.cols; ++x, ++p_x)
		{
			if (std::fabs(*p_x) > 1.0)
			{
				*p_x = (*p_x > 0.0 ? 1.0 : -1.0);
			}
		}
	}
}

void cds::ProxLinfUnitBall(cv::Mat &X)
{
	for (int y = 0; y < X.rows; ++y)
	{
		float *p_x = X.ptr<float>(y);
		
		for (int x = 0; x < X.cols; ++x, ++p_x)
		{
			float normX = std::fabs(*p_x);
			normX = MIN(1.0, normX);

			*p_x /= normX;
		}
	}
}

void cds::ProxL2Inpainting(cv::Mat &X, cv::Mat const &dataTerm, cv::Mat const &mask)
{
	for (int y = 0; y < X.rows; ++y)
	{
		float *p_x = X.ptr<float>(y);
		float const *p_mask = mask.ptr<float>(y);
		float const *p_data = dataTerm.ptr<float>(y);
		
		for (int x = 0; x < X.cols; ++x, ++p_x, ++p_mask, ++p_data)
		{
			if (*p_mask)
			{
				*p_x = *p_data;
			}
		}
	}
}

void cds::ProxLinfUnitBall(cv::Mat &X1, cv::Mat &X2)
{
	int cols = MIN(X1.cols, X2.cols);
	int rows = MIN(X1.rows, X2.rows);
	
	for (int y = 0; y < rows; ++y)
	{
		float *p_x1 = X1.ptr<float>(y);
		float *p_x2 = X2.ptr<float>(y);
		
		for (int x = 0; x < cols; ++x, ++p_x1, ++p_x2)
		{
			float normX = std::sqrt(SQUARED_NORM(*p_x1, *p_x2));
			normX = MAX(1.0, normX);
			
			*p_x1 /= normX;
			*p_x2 /= normX;
		}
	}
}

void cds::ProxInterval(cv::Mat &X, float xmin, float xmax)
{
	for (int y = 0; y < X.rows; ++y)
	{
		float *p_x = X.ptr<float>(y);
		
		for (int x = 0; x < X.cols; ++x, ++p_x)
		{
			*p_x = MIN(MAX(xmin, *p_x), xmax);
		}
	}
}
