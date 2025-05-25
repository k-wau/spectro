# Spectro

A real-time audio visualizer that displays system audio output with a dynamic waveform and frequency-based color transitions.

## Features

- Real-time audio visualization of system output
- Dynamic waveform display with automatic height normalization
- Frequency-based color transitions
- Borderless window design
- Adjustable visualization parameters

## Dependencies

- SFML 2.6+
- PulseAudio
- FFTW3
- CMake 3.0+
- C++11 compatible compiler

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Controls

- `ESC`: Close the application
- `C`: Toggle between colored background and colored waveform
- `Left/Right Arrow`: Adjust waveform length
- `Up/Down Arrow`: Adjust window height

## Performance

The visualizer is optimized for performance with:
- Efficient audio capture using PulseAudio
- Optimized FFT calculations using FFTW3
- Smart buffer management
- Reduced color update frequency
- Efficient vertex array handling

## License

MIT License - See LICENSE file for details 