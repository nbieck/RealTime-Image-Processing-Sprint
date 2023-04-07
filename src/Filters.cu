#include <Filters.h>

#include <GpuUtils.h>
#include <OffsetPointer.h>

__device__ float unnormalizedGaussian(float sigma, int x, int y)
{
	return expf(-(x * x + y * y) / (2 * sigma * sigma));
}

__global__ void blurVertical(const OffsetPointer<uchar3> source, OffsetPointer<uchar3> destination, float sigma)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int kernel_dim = (int)(3 * sigma);

	if (x < destination.width() && y < destination.height())
	{
		float3 accum = make_float3(0, 0, 0);
		float weightAccum = 0;

		for (int i = -kernel_dim; i <= kernel_dim; i++)
		{
			if (y + i >= 0 && y + i < destination.height())
			{
					float gauss = unnormalizedGaussian(sigma, i, 0);
					
					accum = accum + normalizeColor(source(x, y + i)) * gauss;
					weightAccum += gauss;
			}
		}

		destination(x, y) = unnormalizeColor(accum / weightAccum);
	}
}

__global__ void blurHorizontal(const OffsetPointer<uchar3> source, OffsetPointer<uchar3> destination, float sigma)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int kernel_dim = (int)(3 * sigma);

	if (x < destination.width() && y < destination.height())
	{
		float3 accum = make_float3(0, 0, 0);
		float weightAccum = 0;

		for (int j = -kernel_dim; j <= kernel_dim; j++)
		{
			if (x + j >= 0 && x + j < destination.width())
			{
				float gauss = unnormalizedGaussian(sigma, 0, j);
				
				accum = accum + normalizeColor(source(x + j, y)) * gauss;
				weightAccum += gauss;
			}
		}

		destination(x, y) = unnormalizeColor(accum / weightAccum);
	}
}

Image gaussian_blur(Image&& input, float sigma, int rows, int cols)
{
	Image output(input.width(), input.height());

	GpuBuffer<uchar3> d_a(input.width() * input.height());
	GpuBuffer<uchar3> d_b(output.width() * output.height());

	d_a.upload(input.data());

	int cell_width = input.width() / cols;
	int cell_height = input.height() / rows;

	dim3 block(16, 16);

	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			OffsetPointer<uchar3> a_ptr(d_a.ptr(), input.width(), cell_width, cell_height, cell_height * r, cell_width * c);
			OffsetPointer<uchar3> b_ptr(d_b.ptr(), output.width(), cell_width, cell_height, cell_height * r, cell_width * c);

			dim3 grid(divUp(a_ptr.width(), block.x), divUp(a_ptr.height(), block.y));

			blurHorizontal KERNEL_LAUNCH(grid, block) (a_ptr, b_ptr, sigma);
			blurVertical KERNEL_LAUNCH(grid, block) (b_ptr, a_ptr, sigma);
		}
	}

	d_a.download(output.data());

	return output;
}
