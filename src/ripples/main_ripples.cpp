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

#include <CDS.hpp>
#include <cds/dsp/ripples.hpp>

#include <iostream>
#include <string>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cds;

int main(int argc, char * const argv[])
{
	if (argc < 2)
	{
		std::cerr << "Missing image!\n";
		std::cerr << "Usage: " << argv[0] << " anImage [nb_levels_in_DWT]\n";
		return EXIT_FAILURE;
	}

	int levels = 1;
	
	if (argc > 2) levels = atoi(argv[2]);

	// Read an image from the command line
	cv::Mat inputImage = cv::imread(argv[1], 0);
	if (!inputImage.data)
	{
		std::cerr << "Error reading: " << argv[1] << "\n";
		return EXIT_FAILURE;
	}

	cv::Mat inputImage32;
	inputImage.convertTo(inputImage32, CV_32F, 1.0/255.0);

	cv::Mat haar;
	levels = cds::ripples::haar(inputImage32, haar, levels);
	std::cout << "Computed " << levels << " levels.\n";
	RescaleAndDisplay(haar, "Haar");
	
	double inputL2 = cv::norm(inputImage32);
	double haarL2 = cv::norm(haar);

	std::cout << "-->\tL2 norm before DWT:\t" << (inputL2*inputL2) << std::endl;
	std::cout << "-->\tL2 norm after DWT:\t" << (haarL2*haarL2) << std::endl;

	cv::Mat ihaar;
	levels = cds::ripples::ihaar(haar, ihaar, levels);

	double ihaarL2 = cv::norm(ihaar);
	std::cout << "Computed " << levels << " inverse levels.\n";
	RescaleAndDisplay(ihaar, "Haar^-1");

	std::cout << "-->\tL2 norm after IDWT:\t" << (ihaarL2*ihaarL2) << " ?= " << (inputL2*inputL2) << std::endl;

	// Wait
	std::cout << "All done. Press a key to exit!\n";
	cv::waitKey();
	
	// Exit
	return EXIT_SUCCESS;
}
