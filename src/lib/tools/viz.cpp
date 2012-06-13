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

#include <cds/tools/viz.hpp>

#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>

void cds::RescaleAndDisplay(cv::Mat const &anImage, std::string const &displayName)
{
	cv::Mat onscreen;

	switch(anImage.type())
	{
		case CV_8U:
			onscreen = anImage;
			break;
			
		case CV_32F:
			anImage.convertTo(onscreen, CV_8U, 255.0);
			break;
		
		default:
			std::cerr << "Sorry, this image type is not handled!\n";
			return;
	}
	
	cv::imshow(displayName, onscreen);
}

void cds::AdaptiveRescaleAndDisplay(cv::Mat const &anImage, std::string const &displayName)
{
	if (!anImage.data)
	{
		return;
	}
	
	cv::Mat tmp;
	
	if (anImage.type() == CV_8U)
	{
		tmp = anImage;
	}
	else
	{
		cv::normalize(anImage, tmp, 0, 1, CV_MINMAX);
	}
	
	cds::RescaleAndDisplay(tmp, displayName);
}

void cds::DisplayMagnitude(cv::Mat const &twoChannelImage, std::string const &displayName, bool takeLog)
{
	if (twoChannelImage.channels() < 2)
	{
		std::cerr << __FUNCTION__ << "warning: mono-image\n";
		return cds::AdaptiveRescaleAndDisplay(twoChannelImage, displayName);
	}
	
	std::vector<cv::Mat> channels;
	cv::split(twoChannelImage, channels);
	
	cv::Mat magnitude;
	cv::magnitude(channels[0], channels[1], magnitude);

	if (takeLog == true)
	{
		magnitude += 1.0;
		cv::Mat logMagnitude;
		cv::log(magnitude, logMagnitude);
		magnitude = logMagnitude;
	}
	
	double magmin, magmax;
	cv::minMaxLoc(magnitude, &magmin, &magmax);
	
	cds::AdaptiveRescaleAndDisplay(magnitude, displayName);
}

void cds::ComposeSidebySide(cv::Mat const &leftImage, cv::Mat const &rightImage, cv::Mat &composite)
{
	int totalWidth = 2 * MAX(leftImage.cols, rightImage.cols);
	int totalHeight = MAX(leftImage.rows, rightImage.rows);
	
	composite.create(cv::Size(totalWidth, totalHeight), leftImage.type());
	composite.setTo(cv::Scalar::all(0));
	
	cv::Mat targetROI;
	
	int xoffset = -(leftImage.cols - totalWidth/2) / 2;
	int yoffset = -(leftImage.rows - totalHeight) / 2;
	
	targetROI = composite(cv::Rect(xoffset, yoffset, leftImage.cols, leftImage.rows));
	leftImage.copyTo(targetROI);
	
	xoffset = (totalWidth/2) - (rightImage.cols - totalWidth/2)/2;
	yoffset = -(rightImage.rows - totalHeight)/2;
	
	targetROI = composite(cv::Rect(xoffset, yoffset, rightImage.cols, rightImage.rows));
	rightImage.copyTo(targetROI);
}


