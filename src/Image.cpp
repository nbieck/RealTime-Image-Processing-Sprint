#include "Image.h"

#include "thirdparty/stb_image.h"
#include "thirdparty/stb_image_write.h"

#include <iostream>

Image::Image(char const* filename)
{
	int channelsInImage;
	image_data_ = reinterpret_cast<uchar3*>(stbi_load(filename, &width_, &height_, &channelsInImage, 3));
}

Image::Image(int width, int height)
	: width_(width), height_(height)
{
	image_data_ = reinterpret_cast<uchar3*>(malloc(width * height * sizeof(uchar3)));
}

Image::~Image()
{
	stbi_image_free(image_data_);
}

Image::Image(Image&& other) noexcept
	: image_data_(other.image_data_), width_(other.width_), height_(other.height_)
{
	other.image_data_ = nullptr;
}

Image& Image::operator=(Image && other) noexcept
{
	image_data_ = other.image_data_;
	width_ = other.width_;
	height_ = other.height_;
	other.image_data_ = nullptr;

	return *this;
}

uchar3* Image::data()
{
	return image_data_;
}

const uchar3* Image::data() const
{
	return image_data_;
}

void Image::save(const std::string& filename)
{
	std::string extension = filename.substr(filename.length() - 3);
	
	if (extension == "png")
		stbi_write_png(filename.data(), width_, height_, 3, image_data_, 0);
	else if (extension == "jpg")
		stbi_write_jpg(filename.data(), width_, height_, 3, image_data_, 0);
	else if (extension == "bmp")
		stbi_write_bmp(filename.data(), width_, height_, 3, image_data_);
	else
		std::cerr << "unknown extension " << extension << std::endl;
}

int Image::width() const
{
	return width_;
}

int Image::height() const
{
	return height_;
}