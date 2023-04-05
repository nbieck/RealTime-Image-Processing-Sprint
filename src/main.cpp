#include "thirdparty/flags.h"
#include "thirdparty/stb_image_write.h"

#include <iostream>
#include <functional>

#include "Image.h"
#include "Conversion.h"

int main(int argc, char** argv)
{
	flags::args args(argc, argv);

	if (args.positional().size() < 2)
	{
		std::cout << "Usage: " << argv[0] << " inputfile outputfile [-equi_to_cube | -cube_to_equi]" << std::endl;
		std::cout << "Valid output formats: png, jpg, bmp" << std::endl;
		std::cout << "Default format: png" << std::endl;
		std::cout << "Default conversion: equirectangular to cubemap" << std::endl;
		return 0;
	}

	auto input_file_name = args.positional()[0];
	auto output_file_name = args.positional()[1];
	
	bool equiToCube = args.get("equi_to_cube", true);
	bool cubeToEqui = args.get("cube_to_equi", true);

	if (equiToCube == cubeToEqui)
	{
		equiToCube = true;
		cubeToEqui = false;
	}

	std::cout << "Converting " <<
		(equiToCube ? "from equirectangular to cubemap" : "from cubemap to equirectangular") << std::endl;
	std::cout << "Input: " << input_file_name << std::endl;
	std::cout << "Output: " << output_file_name << std::endl;

	Image image(input_file_name.data());

	if (equiToCube)
	{
		image = convertEquirectangularToCubemap(std::move(image));
	}
	else
	{
		image = convertCubemapToEquirectangular(std::move(image));
	}

	image.save(std::string(output_file_name));

	return 0;
}