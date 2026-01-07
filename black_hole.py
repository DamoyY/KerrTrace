import os
import pyglet
from src.renderer import CudaRenderer

current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    window = CudaRenderer(width=2100, height=900, current_dir=current_dir)
    pyglet.app.run()
