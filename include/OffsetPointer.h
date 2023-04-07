#pragma once

// wraps around a pointer and allows accessing based on 0-based 2d indices into a subregion of
// the pointed-to array
// Has no owndership, care must be taken to limit the lifetime to the underlying array
template <typename T>
class OffsetPointer
{
	// continguous 2D data
	T* data_;
	// total width of the 2D array
	int total_width_;
	// offset of the first element of the intended subregion
	int offset_row_, offset_col_;

	// dimension of the subarray
	int width_, height_;

public:

	OffsetPointer(T* data, int total_width, int width, int height, int offset_row = 0, int offset_col = 0)
		: data_(data), total_width_(total_width), offset_row_(offset_row), offset_col_(offset_col), width_(width), height_(height)
	{}

	__device__ __host__ T& operator()(int x, int y)
	{
		int row = y + offset_row_;
		int col = x + offset_col_;

		return data_[row * total_width_ + col];
	}

	__device__ __host__ const T& operator()(int x, int y) const
	{
		int row = y + offset_row_;
		int col = x + offset_col_;

		return data_[row * total_width_ + col];
	}

	__device__ __host__ int width() const
	{
		return width_;
	}

	__device__ __host__ int height() const
	{
		return height_;
	}
};
