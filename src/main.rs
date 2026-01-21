mod blackbody;
mod camera;
mod config;
mod renderer;
use std::{collections::HashSet, num::NonZeroU32, path::Path, sync::Arc, time::Instant};

use anyhow::{Context, Result};
use config::Config;
use glam::Vec3;
use log::{error, info};
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
    keys_pressed: HashSet<KeyCode>,
    mouse_locked: bool,
    last_frame: Instant,
}
impl App {
    fn new(config: Config) -> Self {
        let cam_pos = Vec3::from_array(config.camera.position);
        Self {
            cam_pos,
            cam_yaw: config.camera.yaw,
            cam_pitch: config.camera.pitch,
            fov: config.camera.fov,
            config,
            window: None,
            surface: None,
            renderer: None,
            context: None,
            keys_pressed: HashSet::new(),
            mouse_locked: false,
            last_frame: Instant::now(),
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
        if let (Some(window), Some(renderer), Some(surface)) = (
            self.window.as_ref(),
            self.renderer.as_mut(),
            self.surface.as_mut(),
        ) {
            let (fwd, rgt, up) = camera::calculate_camera_basis(self.cam_yaw, self.cam_pitch);
            let fov_scale = (self.fov.to_radians() / 2.0).tan();
            let buffer_u8 = renderer.render(
                self.cam_pos.to_array(),
                fwd.to_array(),
                rgt.to_array(),
                up.to_array(),
                fov_scale,
            )?;
            if self.config.renderer.save_first_frame {
                let path = Path::new(&self.config.renderer.first_frame_path);
                if !path.exists() {
                    info!("Saving first frame to {:?}", path);
                    image::save_buffer(
                        path,
                        &buffer_u8,
                        self.config.window.width,
                        self.config.window.height,
                        image::ColorType::Rgba8,
                    )?;
                }
            }
            let mut buffer = surface
                .buffer_mut()
                .map_err(|e| anyhow::anyhow!("Failed to access surface buffer: {e}"))?;
            for (i, chunk) in buffer_u8.chunks_exact(4).enumerate() {
                let r = chunk[0] as u32;
                let g = chunk[1] as u32;
                let b = chunk[2] as u32;
                let pixel = (r << 16) | (g << 8) | b;
                if i < buffer.len() {
                    buffer[i] = pixel;
                }
            }
            buffer
                .present()
                .map_err(|e| anyhow::anyhow!("Failed to present frame: {e}"))?;
            window.request_redraw();
        }
        Ok(())
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
                        error!("Failed to initialize renderer: {:?}", e);
                        event_loop.exit();
                    }
                }
                Err(e) => {
                    error!("Failed to create window: {:?}", e);
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
                    error!("Render failed: {:?}", e);
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
                                    let _ = w.set_cursor_visible(true);
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
                        let _ = w.set_cursor_visible(false);
                        let _ = w
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            .or_else(|_| w.set_cursor_grab(winit::window::CursorGrabMode::Locked));
                    }
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
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.mouse_locked {
                    let (dx, dy) = delta;
                    let sensitivity = self.config.controls.mouse_sensitivity as f32; // Assuming config is f32
                    self.cam_yaw += (dx as f32) * sensitivity;
                    self.cam_pitch += (dy as f32) * sensitivity;
                    let min_p = self.config.camera.pitch_limit[0];
                    let max_p = self.config.camera.pitch_limit[1];
                    self.cam_pitch = self.cam_pitch.clamp(min_p, max_p);
                }
            }
            _ => (),
        }
    }
}
fn main() -> Result<()> {
    env_logger::init();
    let config = config::load_config("config.yaml")?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;
    Ok(())
}
