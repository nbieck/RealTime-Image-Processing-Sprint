#pragma once

#include "Image.h"

Image gaussian_blur(Image&& input, float sigma, int rows, int cols);
