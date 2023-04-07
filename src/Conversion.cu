#include "Conversion.h"

#include "GpuUtils.h"
#include "OffsetPointer.h"

#include "math_constants.h"

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

__global__ void equiToCube(const OffsetPointer<uchar3> src, OffsetPointer<uchar3> dst)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int squareDim = dst.height() / 2;

	if (x < dst.width() && y < dst.height())
	{
		float3 cubeCoord = cubeCoordFromXY(x, y, squareDim);

		float3 spherical = cartesianToSpherical(cubeCoord);

		float theta_norm = spherical.x / (CUDART_PI * 2);
		if (theta_norm < 0)
		{
			theta_norm += 1;
		}
		float phi_norm = spherical.y / CUDART_PI;

		dst(x, y) = src((int)(theta_norm * src.width()), (int)(phi_norm * src.height()));
	}
}

__global__ void cubeToEqui(const OffsetPointer<uchar3> src, OffsetPointer<uchar3> dst)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < dst.width() && y < dst.height())
	{
		float3 sphericalCoord = make_float3(((float)x / dst.width()) * CUDART_PI * 2, ((float)y / dst.height()) * CUDART_PI, 2);
		float3 cartesian = sphericalToCartesian(sphericalCoord);

		float2 texCoord = cartesianToTexCoord(cartesian);

		dst(x, y) = src((int)(texCoord.x * src.width()), (int)(texCoord.y * src.height()));
	}
}

Image convertEquirectangularToCubemap(Image&& input, int rows, int cols)
{
	int squareDims = (input.width() / cols) / 4;

	Image result(squareDims * 3 * cols, squareDims * 2 * rows);

	GpuBuffer<uchar3> d_input(input.width() * input.height());
	GpuBuffer<uchar3> d_output(result.width() * result.height());

	d_input.upload(input.data());

	int cell_width_in = input.width() / cols;
	int cell_height_in = input.height() / rows;

	int cell_width_out = result.width() / cols;
	int cell_height_out = result.height() / rows;
	
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			OffsetPointer<uchar3> src_ptr(d_input.ptr(), input.width(), cell_width_in, cell_height_in, r * cell_height_in, c * cell_width_in);
			OffsetPointer<uchar3> dst_ptr(d_output.ptr(), result.width(), cell_width_out, cell_height_out, r * cell_height_out, c * cell_width_out);

			dim3 block(16, 16);
			dim3 grid(divUp(dst_ptr.width(), block.x), divUp(dst_ptr.height(), block.y));

			equiToCube KERNEL_LAUNCH(grid, block) (src_ptr, dst_ptr);
		}
	}

	d_output.download(result.data());

	return result;
}

Image convertCubemapToEquirectangular(Image&& input, int rows, int cols)
{
	int squareDims = (input.width() / cols) / 3;

	Image result(squareDims * 4 * cols, squareDims * 2 * rows);

	GpuBuffer<uchar3> d_input(input.width() * input.height());
	GpuBuffer<uchar3> d_output(result.width() * result.height());

	d_input.upload(input.data());
	
	int cell_width_in = input.width() / cols;
	int cell_height_in = input.height() / rows;

	int cell_width_out = result.width() / cols;
	int cell_height_out = result.height() / rows;
	
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			OffsetPointer<uchar3> src_ptr(d_input.ptr(), input.width(), cell_width_in, cell_height_in, r * cell_height_in, c * cell_width_in);
			OffsetPointer<uchar3> dst_ptr(d_output.ptr(), result.width(), cell_width_out, cell_height_out, r * cell_height_out, c * cell_width_out);

			dim3 block(16, 16);
			dim3 grid(divUp(dst_ptr.width(), block.x), divUp(dst_ptr.height(), block.y));

			cubeToEqui KERNEL_LAUNCH(grid, block) (src_ptr, dst_ptr);
		}
	}

	d_output.download(result.data());

	return result;
}