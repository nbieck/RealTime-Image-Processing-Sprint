#include "thirdparty/flags.h"

#include <iostream>

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
	
	std::optional<bool> equiToCubeOpt = args.get<bool>("equi_to_cube");
	std::optional<bool> cubeToEquiOpt = args.get<bool>("cube_to_equi");

	bool equiToCube = true;
	if (equiToCubeOpt.has_value())
	{
		equiToCube = true;
	}
	else if (cubeToEquiOpt.has_value())
	{
		equiToCube = false;
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