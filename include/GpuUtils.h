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

__device__ float2 operator+(float2 a, float2 b);
__device__ float3 operator/(float3 a, float b);
__device__ int argmax(float3 v);
__device__ float3 abs(float3 v);
__device__ float max(float3 v);
__device__ float3 normalizeColor(uchar3 c);
__device__ uchar3 unnormalizeColor(float3 c);
__device__ float3 operator*(float3 a, float b);
__device__ float3 operator+(float3 a, float3 b);

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