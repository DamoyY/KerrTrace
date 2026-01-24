use std::sync::Arc;

use log::error;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{WindowAttributes, WindowId},
};

use super::App;
use crate::math::f32_from_f64;
impl App {
    pub(super) fn update_fps(&mut self, rendered: bool) {
        if rendered {
            self.fps_frames += 1;
        }
        let elapsed = self.fps_last_instant.elapsed();
        if elapsed >= std::time::Duration::from_secs(1) {
            let elapsed_secs = elapsed.as_secs_f64();
            if self.fps_frames == 0 {
                self.fps_value = 0.0;
            } else {
                let fps = f64::from(self.fps_frames) / elapsed_secs;
                match f32_from_f64(fps) {
                    Ok(value) => self.fps_value = value,
                    Err(err) => {
                        error!("FPS 计算失败: {err}");
                        self.fps_value = 0.0;
                    }
                }
            }
            self.fps_frames = 0;
            self.fps_last_instant = std::time::Instant::now();
        }
    }

    pub(super) fn throttle_if_needed(&mut self) {
        if !self.config.window.vsync {
            return;
        }
        let target = std::time::Duration::from_secs_f32(1.0 / 60.0);
        let elapsed = self.last_present.elapsed();
        if let Some(remaining) = target.checked_sub(elapsed) {
            std::thread::sleep(remaining);
        }
        self.last_present = std::time::Instant::now();
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attr = WindowAttributes::default()
                .with_title("KerrTrace Rust")
                .with_inner_size(winit::dpi::PhysicalSize::new(
                    self.config.window.width,
                    self.config.window.height,
                ));
            match event_loop.create_window(attr) {
                Ok(w) => {
                    self.window = Some(Arc::new(w));
                    if let Err(e) = self.init_renderer() {
                        error!("Failed to initialize renderer: {e:?}");
                        event_loop.exit();
                    }
                }
                Err(e) => {
                    error!("Failed to create window: {e:?}");
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Err(err) = self.update_camera() {
                    error!("Camera update failed: {err:?}");
                    event_loop.exit();
                    return;
                }
                if let Err(e) = self.render() {
                    error!("Render failed: {e:?}");
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let physical_key = event.physical_key;
                if let PhysicalKey::Code(keycode) = physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if keycode == KeyCode::Escape {
                                self.mouse_locked = false;
                                if let Some(w) = &self.window {
                                    w.set_cursor_visible(true);
                                    if let Err(err) =
                                        w.set_cursor_grab(winit::window::CursorGrabMode::None)
                                    {
                                        error!("释放鼠标捕获失败: {err}");
                                    }
                                }
                            }
                            self.keys_pressed.insert(keycode);
                        }
                        ElementState::Released => {
                            self.keys_pressed.remove(&keycode);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed && button == winit::event::MouseButton::Left {
                    self.mouse_locked = true;
                    if let Some(w) = &self.window {
                        w.set_cursor_visible(false);
                        if let Err(err) = w.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            && let Err(fallback_err) =
                                w.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        {
                            error!("鼠标捕获失败: confined={err}, locked={fallback_err}");
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.mouse_locked {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => Ok(y),
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            f32_from_f64(pos.y).map(|value| value / 120.0)
                        }
                    };
                    let scroll = match scroll {
                        Ok(value) => value,
                        Err(err) => {
                            error!("滚轮输入转换失败: {err}");
                            return;
                        }
                    };
                    self.fov -= scroll * self.config.camera.zoom_speed;
                    let min_fov = self.config.camera.fov_limit[0];
                    let max_fov = self.config.camera.fov_limit[1];
                    self.fov = self.fov.clamp(min_fov, max_fov);
                }
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.mouse_locked
        {
            let (dx, dy) = delta;
            let sensitivity = self.config.controls.mouse_sensitivity;
            let dx = match f32_from_f64(dx) {
                Ok(value) => value,
                Err(err) => {
                    error!("鼠标移动 X 轴转换失败: {err}");
                    return;
                }
            };
            let dy = match f32_from_f64(dy) {
                Ok(value) => value,
                Err(err) => {
                    error!("鼠标移动 Y 轴转换失败: {err}");
                    return;
                }
            };
            self.cam_yaw += dx * sensitivity;
            self.cam_pitch -= dy * sensitivity;
            let min_p = self.config.camera.pitch_limit[0];
            let max_p = self.config.camera.pitch_limit[1];
            self.cam_pitch = self.cam_pitch.clamp(min_p, max_p);
        }
    }
}
