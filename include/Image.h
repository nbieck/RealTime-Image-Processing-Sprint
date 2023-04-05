#pragma once

#include <cuda_runtime.h>

#include <string>

class Image
{
	uchar3* image_data_;
	int width_, height_;

public:
	Image(char const* filename);

	Image(int width, int height);

	~Image();

	Image(const Image& other) = delete;

	Image(Image&& other) noexcept;

	Image& operator=(const Image& other) = delete;

	Image& operator=(Image&& other) noexcept;

	uchar3* data();

	const uchar3* data() const;

	void save(const std::string& filename);

	int width() const;

	int height() const;
};
