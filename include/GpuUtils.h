#ifndef GPU_UTILS_H_
#define GPU_UTILS_H_

// workarounds for incorrect highlighting
#include <device_launch_parameters.h>
#ifdef __INTELLISENSE__
	#define KERNEL_LAUNCH(GRID, BLOCK)
#else
	#define KERNEL_LAUNCH(GRID, BLOCK) <<<GRID, BLOCK>>>
#endif

inline int divUp(int a, int b)
{
	return (a + b - 1) / b;
}

__device__ int argmax(const float3& v);
__device__ float3 abs(const float3& v);
__device__ float max(const float3& v);

__device__ float3 normalizeColor(const uchar3& c);
__device__ uchar3 unnormalizeColor(const float3& c);

__device__ float2 operator+(const float2& a, const float2& b);
__device__ float2 operator*(const float2& a, float b);

__device__ float3 operator/(const float3& a, float b);
__device__ float3 operator*(const float3& a, float b);
__device__ float3 operator+(const float3& a, const float3& b);

__device__ float2 normalize(const int2& v);
__device__ int2 operator-(const int2& a, const int2& b);

template <typename T>
class GpuBuffer
{
	T* gpu_data_;
	int elem_count_;

public:

	GpuBuffer(int elemCount)
		: elem_count_(elemCount)
	{
		cudaMalloc(&gpu_data_, elemCount * sizeof(T));
	}

	~GpuBuffer()
	{
		cudaFree(gpu_data_);
	}

	GpuBuffer(const GpuBuffer& other) = delete;

	GpuBuffer(GpuBuffer&& other)
		: gpu_data_(other.gpu_data_), elem_count_(other.elem_count_)
	{
		other.gpu_data_ = nullptr;
	}

	GpuBuffer& operator=(const GpuBuffer& other) = delete;

	GpuBuffer& operator=(GpuBuffer&& other)
	{
		elem_count_ = other.elem_count_;
		gpu_data_ = other.gpu_data_;
		other.gpu_data_ = nullptr;

		return *this;
	}

	void upload(const T* source)
	{
		cudaMemcpy(gpu_data_, source, elem_count_ * sizeof(T), cudaMemcpyHostToDevice);
	}

	void download(T* destination)
	{
		cudaMemcpy(destination, gpu_data_, elem_count_ * sizeof(T), cudaMemcpyDeviceToHost);
	}

	T* ptr()
	{
		return gpu_data_;
	}

	const T* ptr() const
	{
		return gpu_data_;
	}
};

#endif // GPU_UTILS_H_