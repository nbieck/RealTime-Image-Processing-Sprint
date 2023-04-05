#pragma once

#include "Image.h"

/*************************************************

Cubemaps are assumed to be contained in a single image in the following configuration

   ----------------------
   |      |      |      |
   |  -Y  |  +X  |  +Y  |
   |      |      |      |
   ----------------------
   |      |      |      |
   |  -X  |  -Z  |  +Z  |
   |      |      |      |
   ----------------------

Which when put on the cube will unfold as such (right-handed coordinates):

                 --------
                 |      |
                 |  +Z  |
                 |      |
   -----------------------------
   |      |      |      |      |
   |  -X  |  +Y  |  +X  |  -Y  |
   |      |      |      |      |
   -----------------------------
                 |      |
                 |  -Z  |
                 |      |
                 --------

*************************************************/

Image convertEquirectangularToCubemap(Image&& input);

Image convertCubemapToEquirectangular(Image&& input);
