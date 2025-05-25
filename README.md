# spectro

real-time audio visualizer that displays system audio output with frequency-dependent colour coding.

## features

- real-time audio visualization of system output
- dynamic waveform display with automatic height normalization
- frequency-based color transitions
- invert colours support

## dependencies

- SFML 2.6+
- PulseAudio
- FFTW3
- CMake 3.0+
- C++11 compatible compiler

## building

```bash
mkdir build
cd build
cmake ..
make
```

## controls

- `ESC`: close the application
- `C`: toggle between colored background and colored waveform
- `Up/Down Arrow`: adjust window height

## license

MIT License - See LICENSE file for details 
