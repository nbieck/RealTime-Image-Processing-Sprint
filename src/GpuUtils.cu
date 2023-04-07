#include <GpuUtils.h>

#include <cuda_runtime.h>

__device__ float2 operator+(const float2& a, const float2& b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float3 operator/(const float3& a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ int argmax(const float3& v)
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

__device__ float3 abs(const float3& v)
{
	return make_float3(abs(v.x), abs(v.y), abs(v.z));
}

__device__ float max(const float3& v)
{
	return max(v.x, max(v.y, v.z));
}

__device__ float3 normalizeColor(const uchar3& c)
{
	return make_float3(c.x / 255., c.y / 255., c.z / 255.);
}

__device__ uchar3 unnormalizeColor(const float3& c)
{
	return make_uchar3(c.x * 255, c.y * 255, c.z * 255);
}

__device__ float3 operator*(const float3& a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ int2 operator-(const int2& a, const int2& b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}

__device__ float2 normalize(const int2& v)
{
	float length = sqrtf(v.x * v.x + v.y * v.y);

	return make_float2(v.x / length, v.y / length);
}

__device__ float2 operator*(const float2& a, float b)
{
	return make_float2(a.x * b, a.y * b);
}