#include <GpuUtils.h>

#include <cuda_runtime.h>

__device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ int argmax(float3 v)
{
	if (v.x > v.y)
	{
		if (v.x > v.z)
		{
			return 0;
		}
		return 2;
	}
	else
	{
		if (v.y > v.z)
		{
			return 1;
		}
		return 2;
	}
}

__device__ float3 abs(float3 v)
{
	return make_float3(abs(v.x), abs(v.y), abs(v.z));
}

__device__ float max(float3 v)
{
	return max(v.x, max(v.y, v.z));
}

__device__ float3 normalizeColor(uchar3 c)
{
	return make_float3(c.x / 255., c.y / 255., c.z / 255.);
}

__device__ uchar3 unnormalizeColor(float3 c)
{
	return make_uchar3(c.x * 255, c.y * 255, c.z * 255);
}

__device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}