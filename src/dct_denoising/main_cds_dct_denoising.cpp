#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cds.hpp>

void print_usage(char const *commandName)
{
  std::cout << commandName << " anImage [sigmaNoise=0.01]" << std::endl;;
}

int main(int argc, char const *argv[])
{
  // Test command line
  if (argc < 2)
  {
    std::cerr << "Missing argument(s) !\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  // Read an image
  cv::Mat originalImage = cv::imread(argv[1], 0);
  if (!originalImage.data)
  {
    std::cerr << "Error reading: " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat imagef;
  originalImage.convertTo(imagef, CV_32F, 1.0/255.0);
  
  // Optional parameters
  float sigmaNoise = 1e-1;

  if (argc > 2) sigmaNoise = (float)atof(argv[2]);

  // Add noise
  cv::Mat noise(imagef.size(), imagef.type());
  cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(sigmaNoise));

  cv::Mat noisyImage = imagef + noise;

  // Denoise with dct coefficients hard-thresholding
  cv::Mat dct;
  cv::dct(noisyImage, dct);

  double meanCoef = cv::norm(dct) / (dct.rows*dct.cols);

  cv::Mat hardDCT = dct.clone();
  cds::hardThresholding(hardDCT, 3.2*sigmaNoise);
  cv::Mat hardCleanf;
  cv::idct(hardDCT, hardCleanf);
  
  // Denoise wit soft-thresholding
  cv::Mat softDCT = dct.clone();
  cds::softThresholding(softDCT, 1.5*sigmaNoise);
  cv::Mat softCleanf;
  cv::idct(softDCT, softCleanf);
  
  // Show
  cds::RescaleAndDisplay(imagef, "Original image");
  cds::RescaleAndDisplay(noisyImage, "Image + Noise");
  cds::RescaleAndDisplay(hardCleanf, "Hard");
  cds::RescaleAndDisplay(softCleanf, "Soft");

  std::cout << "Noisy image PSNR:\t\t" << cds::PSNR(noisyImage, imagef) << std::endl;

  std::cout << "Reconstruction (hard) PSNR:\t" << cds::PSNR(hardCleanf, imagef) << std::endl;
  std::cout << "Reconstruction (soft) PSNR:\t" << cds::PSNR(softCleanf, imagef) << std::endl;

  std::cout << "---------------------------------------\n";
  std::cout << "Noisy image SNR:\t\t" << cds::SNR(noisyImage, imagef) << std::endl;

  std::cout << "Reconstruction (hard) SNR:\t" << cds::SNR(hardCleanf, imagef) << std::endl;
  std::cout << "Reconstruction (soft) SNR:\t" << cds::SNR(softCleanf, imagef) << std::endl;

  cv::waitKey();

  // Clean-up and exit
  return EXIT_SUCCESS;
}
