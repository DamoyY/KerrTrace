mod blackbody;
mod camera;
mod config;
mod hud;
mod renderer;
use std::{
    collections::HashSet,
    num::NonZeroU32,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use config::Config;
use glam::Vec3;
use hud::{TextStyle, draw_text};
use log::error;
use renderer::CudaRenderer;
use softbuffer::{Context as SoftContext, Surface};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};
struct App {
    config: Config,
    window: Option<Arc<Window>>,
    surface: Option<Surface<Arc<Window>, Arc<Window>>>,
    renderer: Option<CudaRenderer>,
    context: Option<SoftContext<Arc<Window>>>,
    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    fov: f32,
    prev_cam_pos: Vec3,
    prev_cam_yaw: f32,
    prev_cam_pitch: f32,
    prev_fov: f32,
    keys_pressed: HashSet<KeyCode>,
    mouse_locked: bool,
    last_present: Instant,
    last_frame: Option<Vec<u8>>,
    fps_last_instant: Instant,
    fps_frames: u32,
    fps_value: f32,
}
impl App {
    const FOV_EPSILON: f32 = 1e-4;
    const ROTATION_EPSILON: f32 = 1e-4;

    fn new(config: Config) -> Self {
        let cam_pos = Vec3::from_array(config.camera.position);
        let now = Instant::now();
        Self {
            cam_pos,
            cam_yaw: config.camera.yaw,
            cam_pitch: config.camera.pitch,
            fov: config.camera.fov,
            prev_cam_pos: cam_pos,
            prev_cam_yaw: config.camera.yaw,
            prev_cam_pitch: config.camera.pitch,
            prev_fov: config.camera.fov,
            config,
            window: None,
            surface: None,
            renderer: None,
            context: None,
            keys_pressed: HashSet::new(),
            mouse_locked: false,
            last_present: now,
            last_frame: None,
            fps_last_instant: now,
            fps_frames: 0,
            fps_value: 0.0,
        }
    }

    fn init_renderer(&mut self) -> Result<()> {
        let window = self.window.as_ref().context("Window not initialized")?;
        let context = SoftContext::new(window.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create softbuffer context: {e}"))?;
        let mut surface = Surface::new(&context, window.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create softbuffer surface: {e}"))?;
        let size = window.inner_size();
        if let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) {
            surface
                .resize(w, h)
                .map_err(|e| anyhow::anyhow!("Failed to resize surface: {e}"))?;
        }
        self.context = Some(context);
        self.surface = Some(surface);
        let current_dir = std::env::current_dir()?;
        let cuda_dir = current_dir.join("cuda");
        let renderer = CudaRenderer::new(&self.config, &cuda_dir)?;
        self.renderer = Some(renderer);
        Ok(())
    }

    fn update_camera(&mut self) {
        let (fwd, rgt, _up) = camera::calculate_camera_basis(self.cam_yaw, self.cam_pitch);
        let mut speed = self.config.controls.move_speed;
        if self.keys_pressed.contains(&KeyCode::ShiftLeft) {
            speed *= self.config.controls.sprint_multiplier;
        }
        if self.keys_pressed.contains(&KeyCode::KeyW) {
            self.cam_pos += fwd * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            self.cam_pos -= fwd * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            self.cam_pos -= rgt * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            self.cam_pos += rgt * speed;
        }
        if self.keys_pressed.contains(&KeyCode::Space) {
            self.cam_pos.y += speed;
        }
        if self.keys_pressed.contains(&KeyCode::ControlLeft) {
            self.cam_pos.y -= speed;
        }
    }

    fn render(&mut self) -> Result<()> {
        if self.window.is_none() {
            return Ok(());
        }
        if self.renderer.is_none() || self.surface.is_none() {
            return Ok(());
        }
        self.update_render_if_needed()?;
        self.update_fps();
        self.present_frame()?;
        self.throttle_if_needed();
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
        Ok(())
    }

    fn update_render_if_needed(&mut self) -> Result<()> {
        if !self.should_render() {
            return Ok(());
        }
        let (fwd, rgt, up) = camera::calculate_camera_basis(self.cam_yaw, self.cam_pitch);
        let fov_scale = (self.fov.to_radians() / 2.0).tan();
        let buffer_u8 = {
            let renderer = self.renderer.as_mut().context("Renderer not initialized")?;
            renderer.render(
                self.cam_pos.to_array(),
                fwd.to_array(),
                rgt.to_array(),
                up.to_array(),
                fov_scale,
            )?
        };
        if self.config.renderer.save_first_frame {
            let path = Path::new(&self.config.renderer.first_frame_path);
            if !path.exists() {
                image::save_buffer(
                    path,
                    &buffer_u8,
                    self.config.window.width,
                    self.config.window.height,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        self.last_frame = Some(buffer_u8);
        self.prev_cam_pos = self.cam_pos;
        self.prev_cam_yaw = self.cam_yaw;
        self.prev_cam_pitch = self.cam_pitch;
        self.prev_fov = self.fov;
        Ok(())
    }

    fn should_render(&self) -> bool {
        if self.last_frame.is_none() {
            return true;
        }
        let position_delta = (self.cam_pos - self.prev_cam_pos).length();
        if position_delta > self.config.renderer.position_epsilon {
            return true;
        }
        let rotation_changed = (self.cam_yaw - self.prev_cam_yaw).abs() > Self::ROTATION_EPSILON
            || (self.cam_pitch - self.prev_cam_pitch).abs() > Self::ROTATION_EPSILON;
        if rotation_changed {
            return true;
        }
        (self.fov - self.prev_fov).abs() > Self::FOV_EPSILON
    }

    fn present_frame(&mut self) -> Result<()> {
        let buffer_u8 = self.last_frame.as_ref().context("Missing render buffer")?;
        let width = self.config.window.width;
        let height = self.config.window.height;
        let hud_layout = self.build_hud_layout(width, height)?;
        let surface = self.surface.as_mut().context("Surface not initialized")?;
        let mut buffer = surface
            .buffer_mut()
            .map_err(|e| anyhow::anyhow!("Failed to access surface buffer: {e}"))?;
        for (i, chunk) in buffer_u8.chunks_exact(4).enumerate() {
            let pixel =
                (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
            if let Some(slot) = buffer.get_mut(i) {
                *slot = pixel;
            }
        }
        draw_hud(&mut buffer, &hud_layout);
        buffer
            .present()
            .map_err(|e| anyhow::anyhow!("Failed to present frame: {e}"))?;
        Ok(())
    }

    fn build_hud_layout(&self, width: u32, height: u32) -> Result<HudLayout> {
        let margin_x =
            i32::try_from(self.config.hud.margin[0]).context("HUD margin_x exceeds i32 range")?;
        let margin_y =
            i32::try_from(self.config.hud.margin[1]).context("HUD margin_y exceeds i32 range")?;
        let font_size = self.config.hud.font_size;
        if font_size == 0 {
            return Err(anyhow!("HUD 字体大小不能为 0"));
        }
        let scale = font_size.div_ceil(8);
        let scale_i32 = i32::try_from(scale).context("HUD scale exceeds i32 range")?;
        let height_i32 = i32::try_from(height).context("Window height exceeds i32 range")?;
        let line_height = (hud::GLYPH_HEIGHT + 1) * scale_i32;
        let fps_y = height_i32 - margin_y - line_height;
        let style = TextStyle {
            width,
            height,
            color: self.config.hud.color,
            scale,
        };
        let info_text = format!(
            "POS : {:.1} {:.1} {:.1}\nVIEW: Y={:.1} P={:.1} FOV={:.0}",
            self.cam_pos.x, self.cam_pos.y, self.cam_pos.z, self.cam_yaw, self.cam_pitch, self.fov
        );
        let fps_text = format!("FPS: {:.1}", self.fps_value);
        Ok(HudLayout {
            style,
            margin_x,
            margin_y,
            fps_y,
            info_text,
            fps_text,
        })
    }

    fn update_fps(&mut self) {
        self.fps_frames += 1;
        let elapsed = self.fps_last_instant.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let elapsed_secs = elapsed.as_secs_f64();
            let fps = f64::from(self.fps_frames) / elapsed_secs;
            match f32_from_f64(fps) {
                Ok(value) => self.fps_value = value,
                Err(err) => {
                    error!("FPS 计算失败: {err}");
                    self.fps_value = 0.0;
                }
            }
            self.fps_frames = 0;
            self.fps_last_instant = Instant::now();
        }
    }

    fn throttle_if_needed(&mut self) {
        if !self.config.window.vsync {
            return;
        }
        let target = Duration::from_secs_f32(1.0 / 60.0);
        let elapsed = self.last_present.elapsed();
        if let Some(remaining) = target.checked_sub(elapsed) {
            std::thread::sleep(remaining);
        }
        self.last_present = Instant::now();
    }
}

struct HudLayout {
    style: TextStyle,
    margin_x: i32,
    margin_y: i32,
    fps_y: i32,
    info_text: String,
    fps_text: String,
}

fn draw_hud(buffer: &mut [u32], layout: &HudLayout) {
    draw_text(
        buffer,
        &layout.style,
        layout.margin_x,
        layout.margin_y,
        &layout.info_text,
    );
    draw_text(
        buffer,
        &layout.style,
        layout.margin_x,
        layout.fps_y,
        &layout.fps_text,
    );
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
                self.update_camera();
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
fn f32_from_f64(value: f64) -> Result<f32> {
    if !value.is_finite() {
        return Err(anyhow!("数值不是有限值: {value}"));
    }
    let min = f64::from(f32::MIN);
    let max = f64::from(f32::MAX);
    if value < min || value > max {
        return Err(anyhow!("数值超出 f32 范围: {value}"));
    }
    let parsed: f32 = value
        .to_string()
        .parse()
        .map_err(|err| anyhow!("解析 f32 失败: {err}"))?;
    if !parsed.is_finite() {
        return Err(anyhow!("转换后数值不是有限值: {parsed}"));
    }
    Ok(parsed)
}
fn main() -> Result<()> {
    env_logger::init();
    let config = config::load_config("config.yaml")?;
    let event_loop = EventLoop::new()?;
    let control_flow = if config.window.vsync {
        ControlFlow::Wait
    } else {
        ControlFlow::Poll
    };
    event_loop.set_control_flow(control_flow);
    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;
    Ok(())
}
