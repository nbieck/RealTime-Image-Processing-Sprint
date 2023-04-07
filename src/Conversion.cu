#include "Conversion.h"

#include "GpuUtils.h"

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
				// this should be the other way around but for some reason then it doesn't work
				-(y / (float)squareDim - 0.5),
				x / (float)squareDim - 0.5,
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
				y / (float)squareDim - 0.5,
				x / (float)squareDim - 0.5,
				0.5
			);
		}
	}

	return cartCoord;
}

__device__ float lookup(float3 v, int idx)
{
	switch (idx)
	{
	case 0:
		return v.x;
	case 1:
		return v.y;
	default:
		return v.z;
	}
}

enum class CubeFace
{
	PlusX,
	MinX,
	PlusY,
	MinY,
	PlusZ,
	MinZ
};

__device__ CubeFace getCubeFace(float3 coord)
{
	int max_axis = argmax(abs(coord));
	float max_val = lookup(coord, max_axis);

	switch (max_axis)
	{
	case 0:
		if (max_val > 0)
		{
			return CubeFace::PlusX;
		}
		else
		{
			return CubeFace::MinX;
		}
		break;
	case 1:
		if (max_val > 0)
		{
			return CubeFace::PlusY;
		}
		else
		{
			return CubeFace::MinY;
		}
		break;
	case 2:
		if (max_val > 0)
		{
			return CubeFace::PlusZ;
		}
		else
		{
			return CubeFace::MinZ;
		}
		break;
	default:
		return CubeFace::PlusX;
	}
}

__device__ float2 baseCoordFrom3D(float3 coord, CubeFace face)
{
	float2 result = make_float2(0, 0);

	coord = coord / (max(abs(coord)) * 2);

	switch (face)
	{
	case CubeFace::PlusX:
		result = make_float2(
			coord.y + 0.5,
			coord.z + 0.5
		);
		break;
	case CubeFace::MinX:
		result = make_float2(
			-coord.y + 0.5,
			coord.z + 0.5
		);
		break;
	case CubeFace::PlusY:
		result = make_float2(
			-coord.x + 0.5,
			coord.z + 0.5
		);
		break;
	case CubeFace::MinY:
		result = make_float2(
			coord.x + 0.5,
			coord.z + 0.5
		);
		break;
	case CubeFace::PlusZ:
		result = make_float2(
			coord.y + 0.5,
			-coord.x + 0.5
		);
		break;
	case CubeFace::MinZ:
		result = make_float2(
			coord.y + 0.5,
			coord.x + 0.5
		);
		break;
	}

	result.y = 1 - result.y;

	result.x /= 3.;
	result.y /= 2.;

	return result;
}

__device__ float2 offsetFromFace(CubeFace face)
{
	float2 result = make_float2(0, 0);

	switch (face)
	{
	case CubeFace::PlusX:
		result = make_float2(
			1./3.,
			0
		);
		break;
	case CubeFace::MinX:
		result = make_float2(
			0,
			0.5
		);
		break;
	case CubeFace::PlusY:
		result = make_float2(
			2./3.,
			0
		);
		break;
	case CubeFace::MinY:
		result = make_float2(
			0,
			0
		);
		break;
	case CubeFace::PlusZ:
		result = make_float2(
			2./3.,
			0.5
		);
		break;
	case CubeFace::MinZ:
		result = make_float2(
			1./3.,
			0.5
		);
		break;
	}

	return result;
}

// convert from a cartesian coordinate to the corresponsing spot on the cubemap texture
__device__ float2 cartesianToTexCoord(float3 cartesian)
{
	CubeFace face = getCubeFace(cartesian);

	float2 base_coord = baseCoordFrom3D(cartesian, face);
	float2 offset = offsetFromFace(face);

	return base_coord + offset;
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

	if (x < out_width && y < out_height)
	{
		float3 sphericalCoord = make_float3(((float)x / out_width) * CUDART_PI * 2, ((float)y / out_height) * CUDART_PI, 2);
		float3 cartesian = sphericalToCartesian(sphericalCoord);

		float2 texCoord = cartesianToTexCoord(cartesian);

		dst[idx2D(x, y, out_width)] = src[idx2D((int)(texCoord.x * in_width), (int)(texCoord.y * in_height), in_width)];
	}
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
	int squareDims = input.width() / 3;

	Image result(squareDims * 4, squareDims * 2);

	GpuBuffer<uchar3> d_input(input.width() * input.height());
	GpuBuffer<uchar3> d_output(result.width() * result.height());

	d_input.upload(input.data());
	
	dim3 block(16, 16);
	dim3 grid(divUp(result.width(), block.x), divUp(result.height(), block.y));

	cubeToEqui KERNEL_LAUNCH(grid, block) (d_input.ptr(), d_output.ptr(), result.width(), result.height(), input.width(), input.height());

	d_output.download(result.data());

	return result;
}