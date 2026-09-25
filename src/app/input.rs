use super::App;
use crate::config::numbers::f32_from_f64;
use anyhow::Result;
use core::time::Duration;
use log::error;
use std::time::Instant;
use winit::{
    event::{ElementState, MouseScrollDelta},
    keyboard::{KeyCode, PhysicalKey},
};
impl App {
    pub(super) fn update_fps(&mut self, rendered: bool) -> Result<()> {
        if rendered {
            self.fps_frames = self
                .fps_frames
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("FPS 帧计数溢出"))?;
        }
        let elapsed = self.fps_last_instant.elapsed();
        if elapsed < Duration::from_secs(1) {
            return Ok(());
        }
        self.fps_value = if self.fps_frames == 0 {
            0.0_f32
        } else {
            let fps = f64::from(self.fps_frames) / elapsed.as_secs_f64();
            f32_from_f64(fps)?
        };
        self.fps_frames = 0;
        self.fps_last_instant = Instant::now();
        Ok(())
    }
    pub(super) fn capture_mouse(&mut self, locked: bool) -> Result<()> {
        use winit::window::CursorGrabMode;
        let mode = if !locked {
            CursorGrabMode::None
        } else if cfg!(target_os = "windows") {
            CursorGrabMode::Confined
        } else {
            CursorGrabMode::Locked
        };
        if let Some(window) = self.window.as_ref() {
            window.set_cursor_grab(mode)?;
            window.set_cursor_visible(!locked);
        }
        self.mouse_locked = locked;
        Ok(())
    }
    pub(super) fn handle_keyboard_input(&mut self, event: &winit::event::KeyEvent) -> Result<()> {
        let PhysicalKey::Code(keycode) = event.physical_key else {
            return Ok(());
        };
        match event.state {
            ElementState::Pressed => {
                if keycode == KeyCode::Escape {
                    self.capture_mouse(false)?;
                }
                self.keys_pressed.insert(keycode);
            }
            ElementState::Released => {
                self.keys_pressed.remove(&keycode);
            }
        }
        Ok(())
    }
    pub(super) fn handle_mouse_wheel(&mut self, delta: &MouseScrollDelta) {
        let scroll = match *delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(position) => match f32_from_f64(position.y) {
                Ok(value) => value / 120.0_f32,
                Err(error) => {
                    error!("滚轮输入转换失败: {error}");
                    return;
                }
            },
        };
        self.camera.fov = (-scroll).mul_add(self.config.camera.zoom_speed, self.camera.fov);
        self.camera.fov = self.camera.fov.clamp(
            self.config.camera.fov_limit[0],
            self.config.camera.fov_limit[1],
        );
    }
    pub(super) fn handle_mouse_motion(&mut self, delta: (f64, f64)) {
        let (x_delta, y_delta) = delta;
        let x_delta_f32 = match f32_from_f64(x_delta) {
            Ok(value) => value,
            Err(error) => {
                error!("鼠标移动 X 轴转换失败: {error}");
                return;
            }
        };
        let y_delta_f32 = match f32_from_f64(y_delta) {
            Ok(value) => value,
            Err(error) => {
                error!("鼠标移动 Y 轴转换失败: {error}");
                return;
            }
        };
        let sensitivity = self.config.controls.mouse_sensitivity;
        self.camera.yaw = x_delta_f32.mul_add(sensitivity, self.camera.yaw);
        self.camera.pitch = (-y_delta_f32).mul_add(sensitivity, self.camera.pitch);
        self.camera.pitch = self.camera.pitch.clamp(
            self.config.camera.pitch_limit[0],
            self.config.camera.pitch_limit[1],
        );
    }
}
