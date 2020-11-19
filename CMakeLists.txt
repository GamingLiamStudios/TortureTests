cmake_minimum_required(VERSION 3.2)
project(TortureTests)

# Make sure everything is compiled into the same dir
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
option(BUILD_SHARED_LIBS "" off)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2")
else()
    if(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -lstdc++fs -lX11 -lGL -lpthread -lpng")
    elseif(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -luser32 -lgdi32 -lopengl32 -lgdiplus -lShlwapi -ldwmapi -lstdc++fs -static")
    endif()
endif()

add_subdirectory(extern)

set(SOURCE_FILES
    # MAIN
    source/main.cpp

    # CPU Tests
    source/CPU/mandelbrot.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_compile_features(TortureTests PUBLIC cxx_std_17)

find_package(OpenGL REQUIRED)

target_include_directories(TortureTests PUBLIC "extern") 
target_include_directories(TortureTests PUBLIC "source") 
target_include_directories(TortureTests PUBLIC ${OPENGL_INCLUDE_DIRS})

target_link_libraries(TortureTests PRIVATE zlib ${OPENGL_LIBRARIES})
