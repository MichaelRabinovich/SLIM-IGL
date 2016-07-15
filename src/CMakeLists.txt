cmake_minimum_required(VERSION 2.6)
project(SLIM-IGL)

include("../CMakeLists.shared")

find_package(MATLAB REQUIRED)
include_directories( ${MATLAB_INCLUDE_DIR} )

set(SOURCES
  ${PROJECT_SOURCE_DIR}/main.cpp
  ${PROJECT_SOURCE_DIR}/geometric_utils.cpp
  ${PROJECT_SOURCE_DIR}/SLIMData.cpp
  ${PROJECT_SOURCE_DIR}/WeightedGlobalLocal.cpp
  ${PROJECT_SOURCE_DIR}/Linesearch.cpp
  ${PROJECT_SOURCE_DIR}/Slim.cpp
  ${ADDITIONAL_SOURCES}
${YIMG_SOURCES}
)

# Alec: This seems like a clang specific thing and also seems to have an extra
# bracket.
# if(APPLE)
#   set(CMAKE_EXE_LINKER_FLAGS "-rpath ${MATLAB_INCLUDE_DIR}/../../bin/maci64"})
# endif (APPLE) #APPLE

add_executable(${PROJECT_NAME}_bin ${SOURCES} ${SHARED_SOURCES})
#target_link_libraries(${PROJECT_NAME}_bin ${SHARED_LIBRARIES})
target_compile_definitions(${PROJECT_NAME}_bin PRIVATE ${LIBIGL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_bin ${SHARED_LIBRARIES} ${MATLAB_LIBRARIES} ${LIBIGL_LIBRARIES}  ${LIBIGL_VIEWER_EXTRA_LIBRARIES})

#/Users/michaelrabinovich/code_libraries/pardiso-matlab/libpardiso500-MACOS-X86-64.dylib