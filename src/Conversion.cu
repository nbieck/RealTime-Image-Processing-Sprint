#include "Conversion.h"

#include "GpuUtils.h"

// workarounds for incorrect highlighting
#include <device_launch_parameters.h>
#ifdef __INTELLISENSE__
	#define KERNEL_LAUNCH(GRID, BLOCK)
#else
	#define KERNEL_LAUNCH(GRID, BLOCK) <<<GRID, BLOCK>>>
#endif

#include "math_constants.h"

__device__ int idx2D(int x, int y, int width)
{
	return y * width + x;
}

__device__ float3 cartesianToSpherical(float3 cartesian)
{
	float r = sqrtf(cartesian.x * cartesian.x + cartesian.y * cartesian.y + cartesian.z * cartesian.z);

	float theta = atan2f(cartesian.y, cartesian.x);
	float phi = acosf(cartesian.z / r);

	return make_float3(theta, phi, r);
}

__device__ float3 sphericalToCartesian(float3 spherical)
{
	float z = cos(spherical.y) * spherical.z;
	float x = cos(spherical.x) * sin(spherical.y) * spherical.z;
	float y = sin(spherical.x) * sin(spherical.y) * spherical.z;

	return make_float3(x, y, z);
}

__device__ float3 cubeCoordFromXY(int x, int y, int squareDim)
{
	float3 cartCoord = make_float3(1,0,0);

	// identify cube face
	if (x < squareDim)
	{
		if (y < squareDim)
		{
			// -Y
			cartCoord = make_float3(
				x / (float)squareDim - 0.5,
				-0.5,
				-(y / (float)squareDim - 0.5)
			);
		}
		else
		{
			y -= squareDim;
			// -X
			cartCoord = make_float3(
				-0.5,
				-(x / (float)squareDim - 0.5),
				-(y / (float)squareDim - 0.5)
			);
		}
	}
	else if (x < 2 * squareDim)
	{
		x -= squareDim;
		if (y < squareDim)
		{
			// +X
			cartCoord = make_float3(
				0.5,
				x / (float)squareDim - 0.5,
				-(y / (float)squareDim - 0.5)
			);
		}
		else
		{
			y -= squareDim;
			// -Z
			cartCoord = make_float3(
				y / (float)squareDim - 0.5,
				-(x / (float)squareDim - 0.5),
				-0.5
			);
		}
	}
	else
	{
		x -= 2 * squareDim;
		if (y < squareDim)
		{
			// +Y
			cartCoord = make_float3(
				-(x / (float)squareDim - 0.5),
				0.5,
				-(y / (float)squareDim - 0.5)
			);
		}
		else
		{
			y -= squareDim;
			// +Z 
			cartCoord = make_float3(
				x / (float)squareDim - 0.5,
				y / (float)squareDim - 0.5,
				0.5
			);
		}
	}

	return cartCoord;
}

__global__ void equiToCube(const uchar3* src, uchar3* dst, int out_width, int out_height, int in_width, int in_height)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int squareDim = out_height / 2;

	if (x < out_width && y < out_height)
	{
		float3 cubeCoord = cubeCoordFromXY(x, y, squareDim);

		float3 spherical = cartesianToSpherical(cubeCoord);

		float theta_norm = spherical.x / (CUDART_PI * 2);
		if (theta_norm < 0)
		{
			theta_norm += 1;
		}
		float phi_norm = spherical.y / CUDART_PI;

		dst[idx2D(x, y, out_width)] = src[idx2D((int)(theta_norm * in_width), (int)(phi_norm * in_height), in_width)];
	}
}

__global__ void cubeToEqui(const uchar3* src, uchar3* dst, int out_width, int out_height, int in_width, int in_height)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int squareDim = in_height / 2;
}

int divUp(int a, int b)
{
	return (a + b - 1) / b;
}

Image convertEquirectangularToCubemap(Image&& input)
{
	int squareDims = input.width() / 4;

	Image result(squareDims * 3, squareDims * 2);

	GpuBuffer<uchar3> d_input(input.width() * input.height());
	GpuBuffer<uchar3> d_output(result.width() * result.height());

	d_input.upload(input.data());
	
	dim3 block(16, 16);
	dim3 grid(divUp(result.width(), block.x), divUp(result.height(), block.y));

	equiToCube KERNEL_LAUNCH(grid, block) (d_input.ptr(), d_output.ptr(), result.width(), result.height(), input.width(), input.height());

	d_output.download(result.data());

	return result;
}

Image convertCubemapToEquirectangular(Image&& input)
{
	return std::move(input);
}