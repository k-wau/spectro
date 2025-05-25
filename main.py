import pyglet
import sounddevice as sd
import numpy as np
from pyglet.gl import *

# Set up pyglet window
window = pyglet.window.Window(800, 400, caption='Audio Visualizer')

# Audio settings
SAMPLE_RATE = 44100
BLOCK_SIZE = 16
CHANNELS = 1

# Buffer settings
HISTORY_SIZE = 2048  # Number of samples to keep in history
audio_buffer = np.zeros(HISTORY_SIZE)  # Circular buffer to store audio history

# Create vertex lists for upper and lower waveforms
vertices_top = np.zeros(HISTORY_SIZE * 2, dtype=np.float32)
vertices_bottom = np.zeros(HISTORY_SIZE * 2, dtype=np.float32)
vertex_list_top = None
vertex_list_bottom = None

def init_vertex_lists():
    global vertex_list_top, vertex_list_bottom
    
    # Initialize x coordinates
    point_spacing = window.width / (HISTORY_SIZE - 1)
    for i in range(HISTORY_SIZE):
        vertices_top[i*2] = vertices_bottom[i*2] = i * point_spacing
        vertices_top[i*2 + 1] = vertices_bottom[i*2 + 1] = window.height/2

    # Create vertex lists
    vertex_list_top = pyglet.graphics.vertex_list(HISTORY_SIZE,
        ('v2f/stream', vertices_top)
    )
    vertex_list_bottom = pyglet.graphics.vertex_list(HISTORY_SIZE,
        ('v2f/stream', vertices_bottom)
    )

def audio_callback(indata, frames, time, status):
    """Callback function for audio stream"""
    global audio_buffer
    # Roll the buffer to make room for new samples
    # Shift old samples to the left
    audio_buffer = np.roll(audio_buffer, -BLOCK_SIZE)
    # Add new samples at the end
    audio_buffer[-BLOCK_SIZE:] = np.mean(indata, axis=1)

def update_vertex_lists(dt):
    """Update vertex positions based on audio data"""
    if vertex_list_top is None or vertex_list_bottom is None:
        return

    # Update y coordinates only (x coordinates stay fixed)
    for i in range(HISTORY_SIZE):
        vertices_top[i*2 + 1] = window.height/2 + (audio_buffer[i] * window.height/2)
        vertices_bottom[i*2 + 1] = window.height/2 - (audio_buffer[i] * window.height/2)
    
    # Update vertex lists
    vertex_list_top.vertices = vertices_top
    vertex_list_bottom.vertices = vertices_bottom

# Start audio input stream
stream = sd.InputStream(
    channels=CHANNELS,
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    callback=audio_callback
)
stream.start()

@window.event
def on_draw():
    window.clear()
    
    # Enable line smoothing for better appearance
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    
    # Set line color to green
    glColor3f(0.0, 1.0, 0.0)
    
    # Draw the waveforms
    glBegin(GL_LINE_STRIP)
    vertex_list_top.draw(GL_LINE_STRIP)
    glEnd()
    
    glBegin(GL_LINE_STRIP)
    vertex_list_bottom.draw(GL_LINE_STRIP)
    glEnd()

if __name__ == '__main__':
    init_vertex_lists()
    # Schedule updates at 60 FPS
    pyglet.clock.schedule_interval(update_vertex_lists, 1/60.0)
    pyglet.app.run()
