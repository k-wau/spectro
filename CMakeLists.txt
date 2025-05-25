cmake_minimum_required(VERSION 3.10)
project(spectro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SFML
find_package(SFML 2.5 COMPONENTS system window graphics audio REQUIRED)

# Find PulseAudio
find_package(PkgConfig REQUIRED)
pkg_check_modules(PULSEAUDIO REQUIRED libpulse)

# Find FFTW3
pkg_check_modules(FFTW3 REQUIRED fftw3)

# Find Threads
find_package(Threads REQUIRED)

# Add executable
add_executable(spectro main.cpp)

# Include directories
target_include_directories(spectro PRIVATE 
    ${PULSEAUDIO_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIRS}
)

# Link directories
link_directories(${PULSEAUDIO_LIBRARY_DIRS})
link_directories(${FFTW3_LIBRARY_DIRS})

# Link libraries
target_link_libraries(spectro 
    sfml-system
    sfml-window
    sfml-graphics
    sfml-audio
    pulse
    pulse-simple
    fftw3
    Threads::Threads
) 