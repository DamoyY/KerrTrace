import os
import time
import math
import cupy as cp
import numpy as np
import pyglet
from pyglet.window import key
from .camera import calculate_camera_basis
from .blackbody import generate_blackbody_lut


class CudaRenderer(pyglet.window.Window):
    def __init__(self, width=2100, height=900, current_dir="."):
        super().__init__(width=width, height=height, vsync=False)
        self.cam_pos = np.array([80.0, 6.0, 0.0], dtype=np.float32)
        self.cam_yaw = -95.0
        self.cam_pitch = -4.0
        self.fov = 10.0
        self.prev_cam_pos = np.copy(self.cam_pos)
        self.prev_cam_yaw = self.cam_yaw
        self.prev_cam_pitch = self.cam_pitch
        self.prev_fov = self.fov
        self.needs_redraw = True
        self.move_speed = 0.5
        self.mouse_sensitivity = 0.06
        self.block_dim = (16, 16)
        self.spin = 0.25
        self.grid_x = (width + self.block_dim[0] - 1) // self.block_dim[0]
        self.grid_y = (height + self.block_dim[1] - 1) // self.block_dim[1]
        self.grid_dim = (self.grid_x, self.grid_y)
        self.lut, self.lut_max_temp = generate_blackbody_lut()
        self.lut_size = self.lut.shape[0]
        cuda_dir = os.path.join(current_dir, "cuda")
        compile_options = ("-use_fast_math", f"-I{cuda_dir}")
        with open(
            os.path.join(cuda_dir, "kernel.cu"), "r", encoding="utf-8"
        ) as f:
            cuda_source = f.read()
            cuda_source += f"\n//{time.time()}\n"
        self.module = cp.RawModule(code=cuda_source, options=compile_options)
        self.kernel = self.module.get_function("kernel")
        self.channels = 4
        self.image_gpu = cp.zeros(
            (height, width, self.channels), dtype=cp.uint8
        )
        self.nbytes = width * height * self.channels
        self.pinned_mem = cp.cuda.alloc_pinned_memory(self.nbytes)
        self.host_image = np.frombuffer(
            self.pinned_mem, dtype=np.uint8, count=self.nbytes
        ).reshape(height, width, self.channels)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.mouse_locked = False
        self.texture = pyglet.image.Texture.create(
            width, height, internalformat=pyglet.gl.GL_RGBA
        )
        self.info_label = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=15,
            x=10,
            y=10,
            anchor_x="left",
            anchor_y="top",
            width=600,
            multiline=True,
            color=(255, 255, 255, 255),
        )
        self.has_saved_first_frame = False
        self.fps_display = pyglet.window.FPSDisplay(window=self)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.mouse_locked:
            self.cam_yaw += dx * self.mouse_sensitivity
            self.cam_pitch += dy * self.mouse_sensitivity
            self.cam_pitch = max(-80.0, min(80.0, self.cam_pitch))

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.mouse_locked = False
            self.set_exclusive_mouse(False)
            return True
        return super().on_key_press(symbol, modifiers)

    def on_mouse_press(self, x, y, button, modifiers):
        if not self.mouse_locked:
            self.mouse_locked = True
            self.set_exclusive_mouse(True)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.mouse_locked:
            zoom_speed = 5.0
            self.fov -= scroll_y * zoom_speed
            self.fov = max(1.0, min(120.0, self.fov))

    def update_camera(self):
        fwd, rgt, _ = calculate_camera_basis(self.cam_yaw, self.cam_pitch)
        fwd_np = np.array(fwd)
        rgt_np = np.array(rgt)
        speed = self.move_speed
        if self.keys[key.LSHIFT]:
            speed *= 2.0
        if self.keys[key.W]:
            self.cam_pos += fwd_np * speed
        if self.keys[key.S]:
            self.cam_pos -= fwd_np * speed
        if self.keys[key.A]:
            self.cam_pos -= rgt_np * speed
        if self.keys[key.D]:
            self.cam_pos += rgt_np * speed
        if self.keys[key.SPACE]:
            self.cam_pos[1] += speed
        if self.keys[key.LCTRL]:
            self.cam_pos[1] -= speed

    def on_draw(self):
        self.clear()
        self.update_camera()
        position_changed = not np.allclose(
            self.cam_pos, self.prev_cam_pos, atol=1e-5
        )
        rotation_changed = (self.cam_yaw != self.prev_cam_yaw) or (
            self.cam_pitch != self.prev_cam_pitch
        )
        fov_changed = self.fov != self.prev_fov
        should_render = (
            self.needs_redraw
            or position_changed
            or rotation_changed
            or fov_changed
        )
        if should_render:
            fwd, rgt, up = calculate_camera_basis(self.cam_yaw, self.cam_pitch)
            fov_scale = math.tan(math.radians(self.fov) / 2.0)
            self.kernel(
                self.grid_dim,
                self.block_dim,
                (
                    self.image_gpu,
                    cp.int32(self.width),
                    cp.int32(self.height),
                    cp.float32(self.cam_pos[0]),
                    cp.float32(self.cam_pos[1]),
                    cp.float32(self.cam_pos[2]),
                    cp.float32(fwd[0]),
                    cp.float32(fwd[1]),
                    cp.float32(fwd[2]),
                    cp.float32(rgt[0]),
                    cp.float32(rgt[1]),
                    cp.float32(rgt[2]),
                    cp.float32(up[0]),
                    cp.float32(up[1]),
                    cp.float32(up[2]),
                    self.lut,
                    cp.int32(self.lut_size),
                    cp.float32(self.lut_max_temp),
                    cp.float32(fov_scale),
                ),
            )
            self.image_gpu.get(out=self.host_image)
            img_data = pyglet.image.ImageData(
                self.width, self.height, "RGBA", self.host_image.tobytes()
            )
            self.texture.blit_into(img_data, 0, 0, 0)
            self.prev_cam_pos = np.copy(self.cam_pos)
            self.prev_cam_yaw = self.cam_yaw
            self.prev_cam_pitch = self.cam_pitch
            self.prev_fov = self.fov
            self.needs_redraw = False
            info_text = (
                f"POS : {self.cam_pos[0]:.1f} {self.cam_pos[1]:.1f} {self.cam_pos[2]:.1f}\n"
                f"VIEW: Y={self.cam_yaw:.1f} P={self.cam_pitch:.1f} FOV={self.fov:.0f}"
            )
            self.info_label.text = info_text
        self.texture.blit(0, 0)
        self.info_label.y = self.height - 10
        self.info_label.draw()
        self.fps_display.draw()
        if not self.has_saved_first_frame:
            self.texture.save("first_frame.png")
            self.has_saved_first_frame = True
