import os
import time
import math
import cupy as cp
import numpy as np
import pyglet
from pyglet.window import key
from .camera import calculate_camera_basis
from .blackbody import generate_blackbody_lut


def _format_float(value):
    text = f"{float(value):.10g}"
    if "e" in text or "E" in text or "." in text:
        return f"{text}f"
    return f"{text}.0f"


def _build_cuda_defines(kernel_config):
    sky_config = kernel_config["sky"]
    bh_config = kernel_config["black_hole"]
    disk_config = kernel_config["disk"]
    integrator_config = kernel_config["integrator"]
    lines = [
        f"#define CONFIG_SSAA_SAMPLES {int(kernel_config['ssaa_samples'])}",
        f"#define CONFIG_EXPOSURE_SCALE {_format_float(kernel_config['exposure_scale'])}",
        f"#define CONFIG_SKY_GRID_DIVISIONS {int(sky_config['grid_divisions'])}",
        f"#define CONFIG_SKY_LINE_THICKNESS {_format_float(sky_config['line_thickness'])}",
        f"#define CONFIG_SKY_INTENSITY {_format_float(sky_config['intensity'])}",
        f"#define CONFIG_BH_SPIN {_format_float(bh_config['spin'])}",
        f"#define CONFIG_BH_MASS {_format_float(bh_config['mass'])}",
        f"#define CONFIG_DISK_OUTER_RADIUS {_format_float(disk_config['outer_radius'])}",
        f"#define CONFIG_DISK_TEMPERATURE_SCALE {_format_float(disk_config['temperature_scale'])}",
        f"#define CONFIG_INTEGRATOR_INITIAL_STEP {_format_float(integrator_config['initial_step'])}",
        f"#define CONFIG_INTEGRATOR_TOLERANCE {_format_float(integrator_config['tolerance'])}",
        f"#define CONFIG_INTEGRATOR_MAX_STEPS {int(integrator_config['max_steps'])}",
        f"#define CONFIG_INTEGRATOR_MAX_ATTEMPTS {int(integrator_config['max_attempts'])}",
        f"#define CONFIG_TRANSMITTANCE_CUTOFF {_format_float(integrator_config['transmittance_cutoff'])}",
        f"#define CONFIG_HORIZON_EPSILON {_format_float(integrator_config['horizon_epsilon'])}",
        f"#define CONFIG_ESCAPE_RADIUS {_format_float(integrator_config['escape_radius'])}",
    ]
    return "\n".join(lines) + "\n"


class CudaRenderer(pyglet.window.Window):
    def __init__(self, config, current_dir):
        window_config = config["window"]
        camera_config = config["camera"]
        control_config = config["controls"]
        renderer_config = config["renderer"]
        hud_config = config["hud"]
        blackbody_config = config["blackbody"]
        cuda_config = config["cuda"]
        kernel_config = config["kernel"]
        width = window_config["width"]
        height = window_config["height"]
        super().__init__(width=width, height=height, vsync=window_config["vsync"])
        self.cam_pos = np.array(camera_config["position"], dtype=np.float32)
        self.cam_yaw = camera_config["yaw"]
        self.cam_pitch = camera_config["pitch"]
        self.fov = camera_config["fov"]
        self.prev_cam_pos = np.copy(self.cam_pos)
        self.prev_cam_yaw = self.cam_yaw
        self.prev_cam_pitch = self.cam_pitch
        self.prev_fov = self.fov
        self.needs_redraw = True
        self.move_speed = control_config["move_speed"]
        self.sprint_multiplier = control_config["sprint_multiplier"]
        self.mouse_sensitivity = control_config["mouse_sensitivity"]
        self.zoom_speed = camera_config["zoom_speed"]
        self.pitch_limit = camera_config["pitch_limit"]
        self.fov_limit = camera_config["fov_limit"]
        self.position_epsilon = renderer_config["position_epsilon"]
        self.block_dim = tuple(renderer_config["block_dim"])
        self.spin = renderer_config["spin"]
        self.grid_x = (width + self.block_dim[0] - 1) // self.block_dim[0]
        self.grid_y = (height + self.block_dim[1] - 1) // self.block_dim[1]
        self.grid_dim = (self.grid_x, self.grid_y)
        self.lut, self.lut_max_temp = generate_blackbody_lut(
            size=blackbody_config["lut_size"],
            max_temp=blackbody_config["lut_max_temp"],
            wavelength_start=blackbody_config["wavelength_start"],
            wavelength_end=blackbody_config["wavelength_end"],
            wavelength_step=blackbody_config["wavelength_step"],
        )
        self.lut_size = self.lut.shape[0]
        cuda_dir = os.path.join(current_dir, "cuda")
        compile_options = [f"-I{cuda_dir}"]
        if cuda_config["use_fast_math"]:
            compile_options.append("-use_fast_math")
        with open(
            os.path.join(cuda_dir, "kernel.cu"), "r", encoding="utf-8"
        ) as f:
            cuda_source = _build_cuda_defines(kernel_config) + f.read()
            cuda_source += f"\n//{time.time()}\n"
        self.module = cp.RawModule(code=cuda_source, options=tuple(compile_options))
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
        self.hud_margin = tuple(hud_config["margin"])
        self.info_label = pyglet.text.Label(
            "",
            font_name=hud_config["font_name"],
            font_size=hud_config["font_size"],
            x=self.hud_margin[0],
            y=self.hud_margin[1],
            anchor_x=hud_config["anchor_x"],
            anchor_y=hud_config["anchor_y"],
            width=hud_config["width"],
            multiline=True,
            color=tuple(hud_config["color"]),
        )
        self.save_first_frame = renderer_config["save_first_frame"]
        self.first_frame_path = renderer_config["first_frame_path"]
        self.has_saved_first_frame = False
        self.fps_display = pyglet.window.FPSDisplay(window=self)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.mouse_locked:
            self.cam_yaw += dx * self.mouse_sensitivity
            self.cam_pitch += dy * self.mouse_sensitivity
            self.cam_pitch = max(
                self.pitch_limit[0], min(self.pitch_limit[1], self.cam_pitch)
            )

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
            self.fov -= scroll_y * self.zoom_speed
            self.fov = max(self.fov_limit[0], min(self.fov_limit[1], self.fov))

    def update_camera(self):
        fwd, rgt, _ = calculate_camera_basis(self.cam_yaw, self.cam_pitch)
        fwd_np = np.array(fwd)
        rgt_np = np.array(rgt)
        speed = self.move_speed
        if self.keys[key.LSHIFT]:
            speed *= self.sprint_multiplier
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
            self.cam_pos, self.prev_cam_pos, atol=self.position_epsilon
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
        self.info_label.y = self.height - self.hud_margin[1]
        self.info_label.draw()
        self.fps_display.draw()
        if self.save_first_frame and not self.has_saved_first_frame:
            self.texture.save(self.first_frame_path)
            self.has_saved_first_frame = True
