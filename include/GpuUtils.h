#ifndef GPU_UTILS_H_
#define GPU_UTILS_H_

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