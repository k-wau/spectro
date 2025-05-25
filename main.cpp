#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <vector>
#include <deque>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <mutex>
#include <complex>
#include <array>
#include <pulse/pulseaudio.h>
#include <pulse/simple.h>
#include <fftw3.h>

constexpr size_t HISTORY_SIZE = 1024;   // Further reduced for better performance
constexpr size_t SAMPLE_RATE = 44100;
constexpr size_t BLOCK_SIZE = 128;    // Larger blocks for more efficient audio capture
constexpr size_t FFT_SIZE = 1024;     // Smaller FFT size for faster computation
constexpr size_t COLOR_UPDATE_INTERVAL = 3;  // Update colors every N frames

class AudioVisualizer {
private:
    sf::RenderWindow window;
    pa_simple* pa_stream;
    std::deque<float> audioBuffer;
    std::mutex audioBufferMutex;
    bool running;
    std::unique_ptr<std::thread> audioThread;
    
    // Vertex array for drawing
    sf::VertexArray waveform;
    size_t currentHistorySize;
    unsigned int windowHeight;
    size_t frameCount;
    
    // FFT related members
    std::vector<double> fftInput;
    fftw_complex* fftOutput;
    fftw_plan fftPlan;
    sf::Color backgroundColor;
    sf::Color waveformColor;
    bool coloredBackground;
    
    // Helper function to convert frequency to musical note number
    double freqToNoteNumber(double frequency) {
        // A4 = 440Hz, which is note number 69
        return 12 * log2(frequency / 440.0) + 69;
    }
    
    // Helper function to normalize colors to maintain minimum intensity
    sf::Color normalizeColor(uint8_t r, uint8_t g, uint8_t b) {
        const int MIN_TOTAL = 255; // Minimum sum of RGB values
        
        // Calculate current total
        int total = r + g + b;
        
        if (total < MIN_TOTAL) {
            // Need to increase intensity while maintaining color relationships
            double scale = static_cast<double>(MIN_TOTAL) / total;
            r = static_cast<uint8_t>(std::min(255.0, r * scale));
            g = static_cast<uint8_t>(std::min(255.0, g * scale));
            b = static_cast<uint8_t>(std::min(255.0, b * scale));
        }
        
        // Set alpha to 51 (20% opacity, as 255 * 0.2 = 51)
        return sf::Color(r, g, b, 51);
    }
    
    // Convert frequency to color cycling through specific colors
    sf::Color frequencyToColor(double frequency, double intensity) {
        // Define our specific colors with more dramatic transitions
        const std::array<sf::Color, 4> colors = {{
            sf::Color(255, 0, 255),    // Magenta
            sf::Color(0, 0, 255),      // Pure Blue
            sf::Color(0, 255, 255),    // Cyan
            sf::Color(255, 0, 128)     // Hot Pink
        }};
        
        // Map frequency ranges differently
        // Use exponential mapping for more even distribution across frequency range
        double freqFactor = log2(std::max(frequency, 20.0) / 20.0); // Start from 20Hz
        double normalizedFreq = std::min(freqFactor / 10.0, 1.0); // Normalize to 0-1 range
        
        // Map to color cycle position
        double position = normalizedFreq * 4.0; // Map to our 4 colors
        
        // Get the two colors to interpolate between
        size_t colorIndex1 = static_cast<size_t>(position) % 4;
        size_t colorIndex2 = (colorIndex1 + 1) % 4;
        double t = position - floor(position); // Interpolation factor
        
        // Get the two colors
        sf::Color color1 = colors[colorIndex1];
        sf::Color color2 = colors[colorIndex2];
        
        // Use smoothstep interpolation for smoother transitions
        double smoothT = t * t * (3 - 2 * t);
        
        // Interpolate between the colors
        uint8_t r = static_cast<uint8_t>(color1.r * (1 - smoothT) + color2.r * smoothT);
        uint8_t g = static_cast<uint8_t>(color1.g * (1 - smoothT) + color2.g * smoothT);
        uint8_t b = static_cast<uint8_t>(color1.b * (1 - smoothT) + color2.b * smoothT);
        
        // Normalize to maintain vibrancy
        return normalizeColor(r, g, b);
    }
    
    void initFFT() {
        fftInput.resize(FFT_SIZE);
        fftOutput = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (FFT_SIZE/2 + 1));
        
        // Create FFT plan
        fftPlan = fftw_plan_dft_r2c_1d(
            FFT_SIZE,
            fftInput.data(),
            fftOutput,
            FFTW_ESTIMATE
        );
        
        backgroundColor = sf::Color::Black;
    }
    
    void updateColors() {
        // Find the dominant frequency and calculate overall intensity
        double maxMagnitude = 0.0;
        size_t maxIndex = 0;
        
        // Only look at the first half of the FFT output (up to Nyquist frequency)
        for (size_t i = 0; i < FFT_SIZE/2; ++i) {
            double magnitude = sqrt(fftOutput[i][0] * fftOutput[i][0] + 
                                 fftOutput[i][1] * fftOutput[i][1]);
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
                maxIndex = i;
            }
        }

        sf::Color currentColor;
        
        // If there's no significant signal, use black
        if (maxMagnitude < 0.1) {
            currentColor = sf::Color(0, 0, 0, 51);
        } else {
            // Calculate the frequency
            double dominantFreq = static_cast<double>(maxIndex) * SAMPLE_RATE / FFT_SIZE;
            
            // Get base color from frequency
            currentColor = frequencyToColor(dominantFreq, 1.0);
            
            // Add temporal color shift
            static double timeOffset = 0.0;
            timeOffset += 0.005;
            
            // Calculate secondary color
            double secondaryFreq = dominantFreq * 6.0/5.0;
            sf::Color secondaryColor = frequencyToColor(secondaryFreq, 1.0);
            
            // Blend colors
            double blend = 0.8 + 0.2 * sin(timeOffset);
            uint8_t r = static_cast<uint8_t>(currentColor.r * blend + secondaryColor.r * (1 - blend));
            uint8_t g = static_cast<uint8_t>(currentColor.g * blend + secondaryColor.g * (1 - blend));
            uint8_t b = static_cast<uint8_t>(currentColor.b * blend + secondaryColor.b * (1 - blend));
            
            currentColor = normalizeColor(r, g, b);
        }

        // Set colors based on mode
        if (coloredBackground) {
            backgroundColor = currentColor;
            waveformColor = sf::Color::White;
        } else {
            backgroundColor = sf::Color(0, 0, 0, 51);  // Semi-transparent black
            waveformColor = currentColor;
            waveformColor.a = 255;  // Make the line fully opaque
        }
    }
    
    void updateFFT() {
        // Only update colors every N frames
        bool shouldUpdateColors = (frameCount % COLOR_UPDATE_INTERVAL) == 0;
        
        if (!shouldUpdateColors) {
            return;
        }

        std::vector<float> bufferCopy;
        
        // Make a copy of the buffer with the lock
        {
            std::lock_guard<std::mutex> lock(audioBufferMutex);
            bufferCopy.assign(audioBuffer.begin(), audioBuffer.end());
        }
        
        // Copy the most recent FFT_SIZE samples to the FFT input buffer
        size_t offset = std::max(0, static_cast<int>(bufferCopy.size() - FFT_SIZE));
        for (size_t i = 0; i < FFT_SIZE; i += 2) {  // Process every other sample
            // Apply Hanning window
            double window = 0.5 * (1 - cos(2 * M_PI * i / (FFT_SIZE - 1)));
            fftInput[i] = (i + offset < bufferCopy.size()) ? bufferCopy[i + offset] * window : 0.0;
            fftInput[i + 1] = fftInput[i];  // Duplicate the sample for skipped positions
        }
        
        // Perform FFT
        fftw_execute(fftPlan);
        
        updateColors();
    }
    
    std::string getDefaultMonitorSource() {
        pa_mainloop* mainloop = pa_mainloop_new();
        pa_mainloop_api* mainloop_api = pa_mainloop_get_api(mainloop);
        pa_context* context = pa_context_new(mainloop_api, "Monitor Source Finder");
        
        if (pa_context_connect(context, nullptr, PA_CONTEXT_NOFLAGS, nullptr) < 0) {
            pa_context_unref(context);
            pa_mainloop_free(mainloop);
            throw std::runtime_error("Failed to connect to PulseAudio");
        }
        
        std::string monitor_source;
        bool done = false;
        
        // Wait for context to be ready
        while (!done) {
            pa_mainloop_iterate(mainloop, 1, nullptr);
            switch (pa_context_get_state(context)) {
                case PA_CONTEXT_READY:
                    done = true;
                    break;
                case PA_CONTEXT_FAILED:
                case PA_CONTEXT_TERMINATED:
                    pa_context_unref(context);
                    pa_mainloop_free(mainloop);
                    throw std::runtime_error("PulseAudio context failed");
                default:
                    break;
            }
        }
        
        // Get default sink
        pa_operation* op = pa_context_get_server_info(context,
            [](pa_context* c, const pa_server_info* info, void* userdata) {
                if (info && info->default_sink_name) {
                    std::string* monitor_source = static_cast<std::string*>(userdata);
                    *monitor_source = std::string(info->default_sink_name) + ".monitor";
                }
            },
            &monitor_source
        );
        
        if (!op) {
            pa_context_unref(context);
            pa_mainloop_free(mainloop);
            throw std::runtime_error("Failed to get server info");
        }
        
        // Wait for the operation to complete
        while (pa_operation_get_state(op) == PA_OPERATION_RUNNING) {
            pa_mainloop_iterate(mainloop, 1, nullptr);
        }
        
        pa_operation_unref(op);
        pa_context_disconnect(context);
        pa_context_unref(context);
        pa_mainloop_free(mainloop);
        
        if (monitor_source.empty()) {
            throw std::runtime_error("Could not find default monitor source");
        }
        
        return monitor_source;
    }
    
    void initPulseAudio() {
        std::cout << "Initializing PulseAudio..." << std::endl;
        
        std::string monitor_source = getDefaultMonitorSource();
        std::cout << "Using monitor source: " << monitor_source << std::endl;
        
        pa_sample_spec ss;
        ss.format = PA_SAMPLE_S16LE;
        ss.channels = 2;
        ss.rate = SAMPLE_RATE;

        pa_buffer_attr buffer_attr;
        buffer_attr.maxlength = (uint32_t)-1;
        buffer_attr.fragsize = BLOCK_SIZE * 4 * sizeof(int16_t);  // Increased buffer size for smoother capture
        buffer_attr.minreq = (uint32_t)-1;
        buffer_attr.prebuf = (uint32_t)-1;
        buffer_attr.tlength = (uint32_t)-1;

        int error = 0;
        pa_stream = pa_simple_new(
            nullptr,
            "Spectro",
            PA_STREAM_RECORD,
            monitor_source.c_str(),
            "output",
            &ss,
            nullptr,
            &buffer_attr,
            &error
        );

        if (!pa_stream) {
            std::cerr << "Could not create PulseAudio stream: " << pa_strerror(error) << std::endl;
            throw std::runtime_error("Could not create PulseAudio stream: " + 
                std::string(pa_strerror(error)));
        }
        
        std::cout << "PulseAudio initialized successfully" << std::endl;
    }
    
    void audioCapture() {
        std::cout << "Starting audio capture thread..." << std::endl;
        std::vector<int16_t> buffer(BLOCK_SIZE * 2);  // Stereo, so *2
        std::vector<float> tempBuffer;
        
        while (running) {
            int error;
            if (pa_simple_read(pa_stream, buffer.data(), buffer.size() * sizeof(int16_t), &error) < 0) {
                std::cerr << "pa_simple_read failed: " << pa_strerror(error) << std::endl;
                continue;
            }

            // Process audio in a temporary buffer
            tempBuffer.clear();
            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                float left = buffer[i * 2] / 32768.0f;
                float right = buffer[i * 2 + 1] / 32768.0f;
                tempBuffer.push_back((left + right) / 2.0f);
            }
            
            // Update the main buffer with a lock
            {
                std::lock_guard<std::mutex> lock(audioBufferMutex);
                for (float sample : tempBuffer) {
                    if (audioBuffer.size() >= HISTORY_SIZE) {
                        audioBuffer.pop_front();
                    }
                    audioBuffer.push_back(sample);
                }
            }
        }
        std::cout << "Audio capture thread stopping..." << std::endl;
    }
    
    void resizeWindow() {
        window.create(sf::VideoMode(800, windowHeight), "Spectro", sf::Style::None);
        window.setFramerateLimit(60);
    }

    void updateWaveform() {
        std::vector<float> bufferCopy;
        
        // Make a copy of the buffer with the lock
        {
            std::lock_guard<std::mutex> lock(audioBufferMutex);
            bufferCopy.assign(audioBuffer.begin(), audioBuffer.end());
        }
        
        // Ensure currentHistorySize doesn't exceed buffer size
        currentHistorySize = std::min(currentHistorySize, bufferCopy.size());
        
        float pointSpacing = static_cast<float>(window.getSize().x) / (currentHistorySize - 1);
        float centerY = window.getSize().y / 2.0f;
        
        // Find maximum amplitude for normalization
        float maxAmplitude = 0.0001f;  // Small non-zero value to prevent division by zero
        for (size_t i = 0; i < currentHistorySize; i += 2) {  // Skip every other sample for faster processing
            maxAmplitude = std::max(maxAmplitude, std::abs(bufferCopy[i]));
        }
        
        // Calculate scale to use full height
        float scale = centerY / maxAmplitude;
        
        // Update vertex positions
        waveform.resize(currentHistorySize);  // Resize vertex array to match current history size
        for (size_t i = 0; i < currentHistorySize; ++i) {
            float x = i * pointSpacing;
            float y = centerY + (bufferCopy[i] * scale);
            
            waveform[i].position = sf::Vector2f(x, y);
            waveform[i].color = waveformColor;
        }
    }
    
public:
    AudioVisualizer() : 
        window(sf::VideoMode(800, 400), "Spectro", sf::Style::None),
        waveform(sf::LineStrip, HISTORY_SIZE),
        running(true),
        pa_stream(nullptr),
        fftOutput(nullptr),
        coloredBackground(true),
        waveformColor(sf::Color::White),
        backgroundColor(sf::Color::Black),
        currentHistorySize(HISTORY_SIZE),
        windowHeight(400),
        frameCount(0)
    {
        std::cout << "Initializing AudioVisualizer..." << std::endl;
        
        // Initialize audio buffer
        audioBuffer.resize(HISTORY_SIZE, 0.0f);
        
        try {
            // Initialize PulseAudio
            initPulseAudio();
            
            // Initialize FFT
            initFFT();
            
            // Start audio capture thread
            audioThread = std::make_unique<std::thread>(&AudioVisualizer::audioCapture, this);
            
            // Initialize vertex array
            updateWaveform();
            
            std::cout << "AudioVisualizer initialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during initialization: " << e.what() << std::endl;
            throw;
        }
    }
    
    ~AudioVisualizer() {
        std::cout << "Shutting down AudioVisualizer..." << std::endl;
        running = false;
        if (audioThread && audioThread->joinable()) {
            audioThread->join();
        }
        if (pa_stream) {
            pa_simple_free(pa_stream);
        }
        // Clean up FFTW
        if (fftPlan) {
            fftw_destroy_plan(fftPlan);
        }
        if (fftOutput) {
            fftw_free(fftOutput);
        }
        std::cout << "AudioVisualizer shut down successfully" << std::endl;
    }
    
    void run() {
        window.setFramerateLimit(60);
        
        std::cout << "Starting main loop..." << std::endl;
        
        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window.close();
                } else if (event.type == sf::Event::KeyPressed) {
                    if (event.key.code == sf::Keyboard::Escape) {
                        window.close();
                    } else if (event.key.code == sf::Keyboard::C) {
                        coloredBackground = !coloredBackground;
                    } else if (event.key.code == sf::Keyboard::Left) {
                        currentHistorySize = std::max(size_t(256), currentHistorySize / 2);
                    } else if (event.key.code == sf::Keyboard::Right) {
                        currentHistorySize = std::min(HISTORY_SIZE, currentHistorySize * 2);
                    } else if (event.key.code == sf::Keyboard::Up) {
                        windowHeight = std::min(1200u, windowHeight + 100);
                        resizeWindow();
                    } else if (event.key.code == sf::Keyboard::Down) {
                        windowHeight = std::max(200u, windowHeight - 100);
                        resizeWindow();
                    }
                }
            }
            
            updateFFT();
            updateWaveform();
            
            // Draw
            window.clear(backgroundColor);
            window.draw(waveform);
            window.display();
            
            ++frameCount;
        }
        
        std::cout << "Main loop ended" << std::endl;
    }
};

int main() {
    try {
        std::cout << "Starting application..." << std::endl;
        AudioVisualizer visualizer;
        visualizer.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 