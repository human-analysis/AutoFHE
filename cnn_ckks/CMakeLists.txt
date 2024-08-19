cmake_minimum_required(VERSION 3.16)

project(cnn_ckks_project)
set(CMAKE_CXX_COMPILER g++)
set(ENV{SEAL_DIR} /usr/local/lib/cmake/SEAL-3.6)
set(CMAKE_BUILD_TYPE Release)

find_package(SEAL 3.6 REQUIRED)


add_subdirectory(common)
add_subdirectory(cpu-ckks/single-key/cnn)
add_subdirectory(cpu-ckks/single-key/comp)
add_subdirectory(cpu-ckks/single-key/ckks_bootstrapping)
add_executable(cnn)
target_sources(cnn
	PRIVATE
		./run/run_cnn.cpp
		${COMMON_SOURCE_FILES}
		${CNN_SOURCE_FILES}
		${COMP_SOURCE_FILES}
		${BOOTSTRAPPING_SOURCE_FILES}
)
target_compile_options(cnn PRIVATE -w -O0 -std=c++17 -pthread -march=native -fopenmp)
target_link_libraries(cnn PRIVATE ntl gmp m pthread SEAL::seal gomp)
target_include_directories(cnn PUBLIC /usr/include /usr/local/include /usr/local/include/NTL ${COMMON_HEADER_DIR} ${CNN_HEADER_DIR} ${COMP_HEADER_DIR} ${BOOTSTRAPPING_HEADER_DIR})
