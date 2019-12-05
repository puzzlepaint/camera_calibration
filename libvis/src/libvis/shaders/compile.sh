glslangValidator -V -S vert -o shader.vert.spv shader.vert
glslangValidator -V -S frag -o shader.frag.spv shader.frag
../../../build_RelWithDebInfo/bin2c -n vis -H license_header.h shader.vert.spv
../../../build_RelWithDebInfo/bin2c -n vis -H license_header.h shader.frag.spv
