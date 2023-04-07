#include <Image.h>
#include <Filters.h>

#include <thirdparty/flags.h>

#include <iostream>

int main(int argc, char** argv)
{
	auto args = flags::args(argc, argv);

	if (args.positional().size() < 2)
	{
		std::cout << "Usage: " << argv[0] << " input output [-rows R] [-cols C] [-gauss sigma]" << std::endl;
		std::cout << "Supported output formats: png, jpg, bmp" << std::endl;
		std::cout << "Default rows and columns: 1" << std::endl;
		std::cout << "Rows and columns will cause the image to be evenly split and each subdivision will be processed separately" << std::endl;
		std::cout << "-gauss sigma will cause a gaussian blur with a standard deviation of sigma to be applied." << std::endl;
		std::cout << "Only one of the specified filters will be applied." << std::endl;

		return 0;
	}

	auto input = args.positional()[0];
	auto output = args.positional()[1];

	unsigned rows = args.get("rows", 1u);
	unsigned cols = args.get("cols", 1u);

	std::optional<float> sigma = args.get<float>("gauss");

	Image img(input.data());

	if (sigma)
	{
		std::cout << "Blurring image " << input << " with standard deviation " << sigma.value() << " and outputting to " << output << std::endl;
		img = gaussian_blur(std::move(img), sigma.value(), rows, cols);
	}

	img.save(output.data());

	return 0;
}