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

#include <cds/tools/quality.hpp>
#include <opencv2/imgproc/imgproc.hpp>

double cds::SNR(cv::Mat const& testImage, cv::Mat const& gtImage, cv::InputArray const &mask)
{
  double mse = cds::MSE(testImage, gtImage, mask);
  
  double signalPower = cv::norm(gtImage, cv::NORM_L2, mask);
  signalPower *= signalPower;
  int npixels = gtImage.size().area() * gtImage.channels();

  cv::Mat _mask = mask.getMat();
  if (_mask.data)
  {
    npixels = cv::countNonZero(_mask) * gtImage.channels();
  }
  signalPower /= npixels;

  return 20.0*std::log10(signalPower/mse);
}

double cds::PSNR(cv::Mat const &testImage, cv::Mat const &referenceImage, cv::InputArray const &mask, double Imax)
{
  double mse = cds::MSE(testImage, referenceImage, mask);

  return 20.0*std::log10(Imax) - 10.0*std::log(mse);
}

double cds::MSE(cv::Mat const &anImage, cv::Mat const &anotherImage, cv::InputArray const &mask)
{
  CV_Assert(anImage.size() == anotherImage.size() && anImage.type() == anotherImage.type());

  int npixels = anImage.size().area() * anImage.channels();

  cv::Mat _mask = mask.getMat();
  if (_mask.data)
  {
    npixels = cv::countNonZero(_mask) * anImage.channels();
  }

  cv::Mat errorImage = anImage - anotherImage;
  cv::multiply(errorImage, errorImage, errorImage);

  return cv::norm(errorImage, cv::NORM_L1, mask) / npixels;
}

void cds::SSIM(cv::Mat const &X, cv::Mat const &Y, cv::Mat &ssim_map, float L, float k1, float k2, int size)
{
  float const C1 = k1*k1*L*L;
  float const C2 = k2*k2*L*L;

  cv::Mat X2, Y2, XY;
  cv::multiply(X, X, X2);
  cv::multiply(Y, Y, Y2);
  cv::multiply(X, Y, XY);

  cv::Mat mu_x, mu_y;
  cv::boxFilter(X, mu_x, CV_32F, cv::Size(size,size));
  cv::boxFilter(Y, mu_y, CV_32F, cv::Size(size,size));

  cv::Mat mu_x2, mu_y2, mu_xy;
  cv::multiply(mu_x, mu_x, mu_x2);
  cv::multiply(mu_y, mu_y, mu_y2);
  cv::multiply(mu_x, mu_y, mu_xy);

  cv::Mat sigma_x2, sigma_y2, sigma_xy;
  cv::boxFilter(X2, sigma_x2, CV_32F, cv::Size(size,size));
  cv::boxFilter(Y2, sigma_y2, CV_32F, cv::Size(size,size));
  cv::boxFilter(XY, sigma_xy, CV_32F, cv::Size(size,size));
  sigma_x2 -= mu_x2;
  sigma_y2 -= mu_y2;
  sigma_xy -= mu_xy;

  cv::Mat A, B, C, D;

  cv::multiply(mu_x, mu_y, A);
  A *= 2.0;
  A += cv::Scalar::all(C1);

  B = 2.0*sigma_xy;
  B += C2;

  cv::multiply(A, B, ssim_map);

  C = sigma_x2 + sigma_y2;
  C += cv::Scalar::all(C1);

  D = sigma_x2 + sigma_y2;
  D *= C2;

  cv::multiply(C, D, C);

  cv::divide(ssim_map, C, ssim_map);
}
