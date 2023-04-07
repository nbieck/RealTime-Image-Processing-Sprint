#pragma once

#include "Image.h"

Image invokeGaussianBlur(Image&& input, float sigma, int rows, int cols);

Image invokeRadialBlur(Image&& input, float sigma, float2 center, int rows, int cols);

Image invokePixelate(Image&& input, int tileSize, int rows, int cols);
