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

#include "../lib/masking.hpp"
#include "../lib/tv.hpp"
#include "../lib/viz.hpp"
#include "../lib/quality.hpp"

#include <iostream>
#include <string>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cds;

int main(int argc, char * const argv[])
{
	if (argc < 2)
	{
		std::cerr << "Missing image!\n";
		std::cerr << "Usage: " << argv[0] << "[-d -i iterations] anImage\n";
		return EXIT_FAILURE;
	}

	int iterations = 100;
	bool use_diffusion = false;
	bool separate_windows = false;
	
	int option;
	
	while ((option = getopt(argc, argv, "di:s")) != -1)
	{
		switch (option)
		{
		case 'd':
			use_diffusion = true;
			break;
		case 'i':
			iterations = atoi(optarg);
			break;
		case 's':
			separate_windows = true;
			break;
		default:
			break;
		}
	}
	
	// Read an image from the command line
	cv::Mat inputImage = cv::imread(argv[argc-1], 0);
		if (!inputImage.data)
	{
		std::cerr << "Error reading: " << argv[argc-1] << "\n";
		return EXIT_FAILURE;
	}

	cv::Mat inputImage32;
	inputImage.convertTo(inputImage32, CV_32F, 1.0/255.0);

	cv::Size frameSize = inputImage.size();
	
	// Generate various masks
	std::cout << "Generating masks...\n";
	std::vector<cv::Mat> masks;

	cv::Mat randomMask50;
	CreateRandomMask(frameSize, 0.5, randomMask50);
	masks.push_back(randomMask50);
	
	cv::Mat randomMask80;
	CreateRandomMask(frameSize, 0.8, randomMask80);
	masks.push_back(randomMask80);
	
	cv::Mat rowMask;
	CreateInterleavedRowsMask(frameSize, rowMask);
	masks.push_back(rowMask);
	
	cv::Rect rectangle;
	float npix = 0.5*inputImage.size().area();
	npix = std::sqrt(npix);
	int radius = (int)std::floor(npix);
	rectangle = cv::Rect(MAX(0, (inputImage.cols-radius)/2), MAX(0, (inputImage.rows-radius)/2),
						 MIN(radius, (inputImage.cols-radius)/2), MIN(radius, (inputImage.rows-radius)/2));
	cv::Mat rectMask;
	CreateRectangularMask(frameSize, rectangle, rectMask);
	masks.push_back(rectMask);
	
	// Generate the corresponding masked images
	std::cout << "Masking images...\n";
	std::vector<cv::Mat> maskedInputs(masks.size());
	for (int i = 0; i < masks.size(); ++i)
	{
		cv::multiply(inputImage32, masks[i], maskedInputs[i]);
	}
	
	// For each image, reconstruct it
	std::cout << "Reconstruction...\n";
	std::vector<cv::Mat> reconstructionResults(masks.size());
	for (int i = 0; i < masks.size(); ++i)
	{
		// Diffuse
		if (use_diffusion)
		{
			TvDiffusion(maskedInputs[i], reconstructionResults[i], iterations, 10);
		}
		else
		{
			TvInpainting(maskedInputs[i], masks[i], reconstructionResults[i], iterations);
		}
	}
	
	// SNR measures 
	std::vector<double> snr_before;
	std::vector<double> snr_after;
	
	for (int i = 0; i < maskedInputs.size(); ++i)
	{
		snr_before.push_back(cds::SNR(maskedInputs[i], inputImage32));
		snr_after.push_back(cds::SNR(reconstructionResults[i], inputImage32));
	}
	
	// OK, time for show off !
	std::cout << "All done, displaying...\n";
	
	for (int i = 0; i < maskedInputs.size(); ++i)
	{
		std::cout << "Masked SNR = " << snr_before[i];
		std::cout << "\tReconstruction SNR = " << snr_after[i];
		std::cout << std::endl;
	}
	
	if (separate_windows == false)
	{
		cv::Mat composite;
	
		ComposeSidebySide(maskedInputs[0], reconstructionResults[0], composite);
		RescaleAndDisplay(composite, "Random 50");
	
		ComposeSidebySide(maskedInputs[1], reconstructionResults[1], composite);
		RescaleAndDisplay(composite, "Random 80");

		ComposeSidebySide(maskedInputs[2], reconstructionResults[2], composite);
		RescaleAndDisplay(composite, "Lines");

		ComposeSidebySide(maskedInputs[3], reconstructionResults[3], composite);
		RescaleAndDisplay(composite, "Rectangle mask");	
	}
	else
	{
		RescaleAndDisplay(maskedInputs[0], "Random 50");
		RescaleAndDisplay(reconstructionResults[0], "Inpainting 50");

		RescaleAndDisplay(maskedInputs[1], "Random 80");
		RescaleAndDisplay(reconstructionResults[1], "Inpainting 80");

		RescaleAndDisplay(maskedInputs[2], "Lines");
		RescaleAndDisplay(reconstructionResults[2], "Lines-Inpainted");

		RescaleAndDisplay(maskedInputs[3], "Rectangle mask");
		RescaleAndDisplay(reconstructionResults[3], "Rectangle Inpainting");
	}
	
	// Wait
	std::cout << "All done. Press a key to exit!\n";
	cv::waitKey();
	
	// Exit
	return EXIT_SUCCESS;
}
