#include <Image.h>
#include <Filters.h>

#include <thirdparty/flags.h>

#include <iostream>

int main(int argc, char** argv)
{
	auto args = flags::args(argc, argv);

	if (args.positional().size() < 2)
	{
		std::cout << "Usage: " << argv[0] << " input output [-rows R] [-cols C] [-gauss sigma] [-radial sigma [-center_x X] [-center_y Y]] [-pixel size]" << std::endl;
		std::cout << "Supported output formats: png, jpg, bmp" << std::endl;
		std::cout << "Default rows and columns: 1" << std::endl;
		std::cout << "Rows and columns will cause the image to be evenly split and each subdivision will be processed separately" << std::endl;
		std::cout << "-gauss sigma will cause a gaussian blur with a standard deviation of sigma to be applied." << std::endl;
		std::cout << "-radial sigma applies a radial blur with the given standard deviation" << std::endl;
		std::cout << "    By default it is centered in the middle of each subdivision, -center_x and -center_y can be used to change this (coordinates are normalized with (0,0) in the top left)" << std::endl;
		std::cout << "-pixel size pixelates the image with a tile size of size" << std::endl;
		std::cout << "Only one of the specified filters will be applied." << std::endl;

		return 0;
	}

	auto input = args.positional()[0];
	auto output = args.positional()[1];

	unsigned rows = args.get("rows", 1u);
	unsigned cols = args.get("cols", 1u);

	std::optional<float> gaussian = args.get<float>("gauss");
	std::optional<float> radial = args.get<float>("radial");
	std::optional<int> pixel = args.get<int>("pixel");

	Image img(input.data());

	if (gaussian)
	{
		std::cout << "Blurring image " << input << " with standard deviation " << gaussian.value() << " and outputting to " << output << std::endl;
		img = invokeGaussianBlur(std::move(img), gaussian.value(), rows, cols);
	}
	else if (radial)
	{
		float2 center = make_float2(args.get("center_x", 0.5f), args.get("center_y", 0.5f));

		std::cout << "Radial blur on image " << input << " writing to " << output << std::endl;
		std::cout << "Centered on (" << center.x << ", " << center.y << ") with standard deviation of " << radial.value() << " pixels" << std::endl;
		img = invokeRadialBlur(std::move(img), radial.value(), center, rows, cols);
	}
	else if (pixel)
	{
		std::cout << "Pixelating image" << input << " writing to " << output << " with tile size " << pixel.value() << std::endl;

		img = invokePixelate(std::move(img), pixel.value(), rows, cols);
	}

	img.save(output.data());

	return 0;
}