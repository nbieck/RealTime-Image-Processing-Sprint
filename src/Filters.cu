#include <Filters.h>

#include <GpuUtils.h>
#include <OffsetPointer.h>

__device__ float unnormalizedGaussian(float sigma, int x, int y)
{
	return expf(-(x * x + y * y) / (2 * sigma * sigma));
}

__global__ void blur(const OffsetPointer<uchar3> source, OffsetPointer<uchar3> destination, float sigma)
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
				for (int j = -kernel_dim; j <= kernel_dim; j++)
				{
					if (x + j >= 0 && x + j < destination.width())
					{
						float gauss = unnormalizedGaussian(sigma, i, j);
						
						accum = accum + normalizeColor(source(x + j, y + i)) * gauss;
						weightAccum += gauss;
					}
				}
			}
		}

		destination(x, y) = unnormalizeColor(accum / weightAccum);
	}
}

Image gaussian_blur(Image&& input, float sigma, int rows, int cols)
{
	Image output(input.width(), input.height());

	GpuBuffer<uchar3> d_input(input.width() * input.height());
	GpuBuffer<uchar3> d_output(output.width() * output.height());

	d_input.upload(input.data());

	int cell_width = input.width() / cols;
	int cell_height = input.height() / rows;

	dim3 block(16, 16);
	dim3 grid(divUp(output.width(), block.x), divUp(output.height(), block.y));

	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			OffsetPointer<uchar3> input_ptr(d_input.ptr(), input.width(), cell_width, cell_height, cell_height * r, cell_width * c);
			OffsetPointer<uchar3> output_ptr(d_output.ptr(), output.width(), cell_width, cell_height, cell_height * r, cell_width * c);

			blur KERNEL_LAUNCH(grid, block) (input_ptr, output_ptr, sigma);
		}
	}

	d_output.download(output.data());

	return output;
}
