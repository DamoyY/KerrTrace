import os
import pyglet
from src.config import load_config
from src.renderer import CudaRenderer

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(current_dir, "config.yaml"))
    window = CudaRenderer(
        config=config,
        current_dir=current_dir,
    )
    pyglet.app.run()
