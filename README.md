# ComputersDontSee #

Emmanuel d'Angelo

http://www.computersdontsee.net

[@sansuiso](http://twitter.com/sansuiso)

sansuiso@computersdontsee.net

## Overview ##

The code presented here is a companion to [my website](http://www.computersdontsee.net).
My goal is to provide some example code and tutorial-style explanations about variational image processing techniques.

While most of the related resources use matlab code, my goal is to provide C++ examples instead.

## License ##

Modified BSD-style (see file LICENCE.txt)

## Technical requirements ##

- OpenCV 2.4, C++ API (but should work with previous 2.x C++ versions)
- cmake to build the library and the executables

## General overview ##

The code repository is divided in two parts : a library and some command line applications.

## Available functions ##

### Image denoising  ###

- **Rudin-Osher-Fatemi (TV-L2) denoising**
Implemented using algorithm 1 of [Ref. 1][1], i.e. primal-dual first order scheme without acceleration.

### Image inpainting ###

- **TV constrained inpainting**

## References ##

[1]: Chambolle, A., Pock, T. (2010). A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging. Journal of Mathematical Imaging and Vision, 40(1), 120–145.