#include <cds/dsp/ripples.hpp>
#include <iostream>

float const kHaarNormFactorS = sqrtf(2.0);
float const kHaarNormFactorD = 1.0 / sqrtf(2.0);

#define HAAR_NORMFACTOR_S 1.4142135623731
#define HAAR_NORMFACTOR_D 0.70710678118655

#define SQRT2 1.4142135623731
#define INV_SQRT2 0.70710678118655
#define SQRT3 1.73205080756888
#define INV_SQRT3 0.57735026918963

//-----------------------------
// Local functions declarations
//-----------------------------
void haar_h(cv::Mat const &srcRow, float *s, float *d);
void haar_v(cv::Mat const &srcCol, float *s, float *d);
void ihaar_h(cv::Mat const &srcRow, float *dest);
void ihaar_v(cv::Mat const &srcCol, float *dest);
void d4_h(cv::Mat const &srcRow, float *s, float *d);
void d4_v(cv::Mat const &srcCol, float *s, float *d);
void id4_h();
void id4_v();
void cdf46_h(cv::Mat const &srcRow, float *s, float *d);
void cdf46_v(cv::Mat const &srcCol, float *s, float *d);
void icdf46_h();
void icdf46_v();

//-----------------------------
// Public Implementations
//-----------------------------
int cds::ripples::haar(cv::Mat const &anImage, cv::Mat &analysis, int maxLevels, int direction)
{
  int levels = 0;

  int currentWidth = anImage.cols;
  int currentHeight = anImage.rows;

  analysis = anImage.clone();

  while ( (currentWidth % 2 == 0) && (currentHeight % 2 == 0) &&
	  (currentWidth > 0) && (currentHeight > 0) && (levels < maxLevels) )
  {
    // Get source and target ROI's
    cv::Mat dest = analysis(cv::Rect(0, 0, currentWidth, currentHeight));

    float *s = new float[currentWidth/2];
    float *d = new float[currentWidth/2];
    // Proceed along the rows
    for (int y = 0; y < currentHeight; ++y)
    {
      cv::Mat const srcRow = dest.row(y);
      haar_h(srcRow, s, d);

      float *p_dest = dest.ptr<float>(y);
      memcpy(p_dest, s, 4*currentWidth/2);
      p_dest += (currentWidth/2);
      memcpy(p_dest, d, 4*currentWidth/2);
    }
    delete[] s;
    delete[] d;

    // Proceed along the columns
    s = new float[currentHeight/2];
    d = new float[currentHeight/2];
    for (int x = 0; x < currentWidth; ++x)
    {
      cv::Mat const srcCol = dest.col(x);
      haar_v(srcCol, s, d);
      
      for (int y = 0; y < currentHeight/2; ++y)
      {
	float *p_s = dest.ptr<float>(y) + x;
	float *p_d = dest.ptr<float>(y+ (currentHeight/2)) + x;
	*p_d = *(d+y);
	*p_s = *(s+y);
      }
    }
    delete[] s;
    delete[] d;

    // Next
    ++levels;
    currentHeight /= 2;
    currentWidth /= 2;
  }

  return levels;
}

int cds::ripples::ihaar(cv::Mat const &coefficients, cv::Mat &synthesis, int maxLevels)
{
  int levels = 0;

  synthesis = coefficients.clone();

  int currentWidth = coefficients.cols;
  int currentHeight = coefficients.rows;

  levels = maxLevels - 1;
  while (levels > 0)
  {
    currentWidth /= 2;
    currentHeight /= 2;
    --levels;
  }

  levels = 0;
  while (levels < maxLevels)
  {
    cv::Mat src = synthesis(cv::Rect(0, 0, currentWidth, currentHeight));

    // Proceed along the columns
    float *dest = new float[currentHeight];
    std::cerr << "Vertical\n";
    for (int x = 0; x < currentWidth; ++x)
    {
      cv::Mat srcCol = src.col(x);
      ihaar_v(srcCol, dest);

      for (int y = 0; y < currentHeight; ++y)
      {
	*(synthesis.ptr<float>(y)+x) = *(dest+y);
      }
    }
    delete[] dest;

    // Proceed along the rows
    std::cerr << "Horizontal\n";
    dest = new float[currentWidth];
    for (int y = 0; y < currentHeight; ++y)
    {
      cv::Mat srcRow = src.row(y);
      ihaar_h(srcRow, dest);
      memcpy(synthesis.ptr<float>(y), dest, 4*currentWidth);
    }
    delete[] dest;

    // Next level
    currentWidth *= 2;
    currentHeight *= 2;
    ++levels;
  }

  return levels;
}

int cds::ripples::daubechies4(cv::Mat const &anImage, cv::Mat &analysis, int maxLevels, int direction)
{
  int levels = 0;

  return levels;
}

int cds::ripples::idaubechies4(cv::Mat const &coefficients, cv::Mat &synthesis, int maxLevels)
{
  int levels = 0;

  return levels;
}

int cds::ripples::cdf46(cv::Mat const &anImage, cv::Mat &analysis, int maxLevels, int direction)
{
  int levels = 0;

  return levels;
}

int cds::ripples::icdf46(cv::Mat const &coefficients, cv::Mat &synthesis, int maxLevels)
{
  int levels = 0;

  return levels;
}

void haar_h(cv::Mat const &srcRow, float *s, float *d)
{
  int N = srcRow.cols / 2;

  float const *p_src = srcRow.ptr<float>(0);

  // Loop over the remains of the row
  for (int n = 0; n < N; ++n)
  {
    // Coefficients
    float a = *p_src++;
    float b = *p_src++;

    *s++ = ((a+b)*INV_SQRT2);
    *d++ = ((a-b)*INV_SQRT2);
  }
}

 void ihaar_h(cv::Mat const &srcRow, float *dest)
{
  int N = srcRow.cols/2;

  float const *s = srcRow.ptr<float>(0);
  float const *d = srcRow.ptr<float>(0) + N;

  for (int n = 0; n < N; ++n)
  {
    float s1 = *s++ * INV_SQRT2;
    float d1 = *d++ *INV_SQRT2;

    *dest = s1 + d1;
    *(dest+1) = s1 - d1;

    dest += 2;
  }
}

void haar_v(cv::Mat const &srcCol, float *s, float *d)
{
  int N = srcCol.rows / 2;

  int srcStep = srcCol.step1();

  float const *p_src = srcCol.ptr<float>(0);

  for (int n = 0; n < N; ++n)
  {
    // Coefficients
    float a = *srcCol.ptr<float>(2*n);
    float b = *srcCol.ptr<float>(2*n+1);
    
    *s++ = ((a+b)*INV_SQRT2);
    *d++ = ((a-b)*INV_SQRT2);
  }
  
  return;
}

void ihaar_v(cv::Mat const &srcCol, float *dest)
{
  int N = srcCol.rows / 2;
  int srcStep = srcCol.step1();

  float const *s = srcCol.ptr<float>(0);
  float const *d = s + N*srcStep;

  for (int n = 0; n < N; ++n)
  {
    float d1 = *d * INV_SQRT2;
    float s1 = *s * INV_SQRT2;

    *dest = s1 + d1;
    *(dest+1) = s1 - d1;
    
    d += srcStep;
    s += srcStep;
    dest += 2;
  }
}

void d4_h(cv::Mat const &srcRow, float *s, float *d)
{
  int N = srcRow.cols;

  float const *p_src = srcRow.ptr<float>(0);
  for (int n = 0; n < N/2; ++n)
  {
    *(s+n) = *p_src + SQRT3*(*(p_src+1));
    p_src += 2;
  }

  p_src = srcRow.ptr<float>(0);
  *d = *(p_src+1) - SQRT3*0.25*(*s) - (SQRT3-2.0)*0.25*(*(s+N/2-1));
  p_src += 2;
  for (int n = 1; n < N/2; ++n)
  {
    *(d+n) = *(p_src+1) - SQRT3*0.25*(*(s+n)) - (SQRT3-2.0)*0.25*(*(s+n-1));
    p_src += 2;
  }

  for (int n = 0; n < N/2-1; ++n)
  {
    *(s+n) -= *(d+n+1);
  }
  *(s+N/2-1) -= *d;

  for (int n = 0; n < N/2; ++n)
  {
    *(s+n) *= (SQRT3-1.0)*INV_SQRT2;
    *(d+n) *= (SQRT3+1.0)*INV_SQRT2;
  }
}

void id4_h()
{
}

void d4_v(cv::Mat const &srcCol, float *s, float *d)
{
}

void id4_v()
{
}

void cdf46_h(cv::Mat const &srcRow, float *s, float *d)
{
}

void icdf46_h()
{
}

void cdf46_v(cv::Mat const &srcCol, float *s, float *d)
{
}

void icdf46_v()
{
}
