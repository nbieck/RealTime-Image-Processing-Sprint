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

__global__ void radialBlur(const OffsetPointer<uchar3> source, OffsetPointer<uchar3> destination, float sigma, int2 center)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < destination.width() && y < destination.height())
	{
		int kernel_dim = (int)(3 * sigma);

		int2 offset = make_int2(x, y) - center;
		float2 normOffset = normalize(offset);
		float aspectRatio = (float)destination.width() / destination.height();

		float3 accum = make_float3(0, 0, 0);
		float weightAccum = 0;

		for (int i = -kernel_dim; i <= kernel_dim; i++)
		{
			float2 lookupOffset = normOffset * i;
			//lookupOffset.y *= aspectRatio;

			if (x + lookupOffset.x >= 0 && x + lookupOffset.x < destination.width()
				&& y + lookupOffset.y >= 0 && y + lookupOffset.y < destination.height())
			{
				float gauss = unnormalizedGaussian(sigma, i, 0);

				accum = accum + normalizeColor(source(x + lookupOffset.x, y + lookupOffset.y)) * gauss;
				weightAccum += gauss;
			}
		}

		destination(x, y) = unnormalizeColor(accum / weightAccum);
	}
}

__global__ void pixelateHorizontal(const OffsetPointer<uchar3> source, OffsetPointer<uchar3> destination, int tileSize)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (y < destination.height() && x * tileSize < destination.width())
	{
		float3 accum = make_float3(0, 0, 0);
		int count = 0;

		for (int i = 0; i < tileSize; ++i)
		{
			int pix_x = x * tileSize + i;
			if (pix_x < destination.width())
			{
				accum = accum + normalizeColor(source(pix_x, y));
				count++;
			}
		}

		uchar3 color = unnormalizeColor(accum / count);

		for (int i = 0; i < tileSize; ++i)
		{
			int pix_x = x * tileSize + i;
			if (pix_x < destination.width())
			{
				destination(pix_x, y) = color;
			}
		}
	}
}

__global__ void pixelateVertical(const OffsetPointer<uchar3> source, OffsetPointer<uchar3> destination, int tileSize)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (y * tileSize < destination.height() && x < destination.width())
	{
		float3 accum = make_float3(0, 0, 0);
		int count = 0;

		for (int i = 0; i < tileSize; ++i)
		{
			int pix_y = y * tileSize + i;
			if (pix_y < destination.height())
			{
				accum = accum + normalizeColor(source(x, pix_y));
				count++;
			}
		}

		uchar3 color = unnormalizeColor(accum / count);

		for (int i = 0; i < tileSize; ++i)
		{
			int pix_y = y * tileSize + i;
			if (pix_y < destination.height())
			{
				destination(x, pix_y) = color;
			}
		}
	}
}

Image invokeGaussianBlur(Image&& input, float sigma, int rows, int cols)
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

Image invokeRadialBlur(Image&& input, float sigma, float2 center, int rows, int cols)
{
	Image output(input.width(), input.height());

	GpuBuffer<uchar3> d_a(input.width() * input.height());
	GpuBuffer<uchar3> d_b(output.width() * output.height());

	d_a.upload(input.data());

	int cell_width = input.width() / cols;
	int cell_height = input.height() / rows;

	dim3 block(16, 16);

	int2 center_pix = make_int2((int)(center.x * cell_width), (int)(center.y * cell_height));

	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			OffsetPointer<uchar3> a_ptr(d_a.ptr(), input.width(), cell_width, cell_height, cell_height * r, cell_width * c);
			OffsetPointer<uchar3> b_ptr(d_b.ptr(), output.width(), cell_width, cell_height, cell_height * r, cell_width * c);

			dim3 grid(divUp(a_ptr.width(), block.x), divUp(a_ptr.height(), block.y));

			radialBlur KERNEL_LAUNCH(grid, block) (a_ptr, b_ptr, sigma, center_pix);
		}
	}

	d_b.download(output.data());

	return output;
}

Image invokePixelate(Image&& input, int tileSize, int rows, int cols)
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

			dim3 gridH(divUp(a_ptr.width() / tileSize, block.x), divUp(a_ptr.height(), block.y));
			dim3 gridV(divUp(a_ptr.width(), block.x), divUp(a_ptr.height() / tileSize, block.y));

			pixelateHorizontal KERNEL_LAUNCH(gridH, block) (a_ptr, b_ptr, tileSize);
			pixelateVertical KERNEL_LAUNCH(gridV, block) (b_ptr, a_ptr, tileSize);
		}
	}

	d_b.download(output.data());

	return output;
}
