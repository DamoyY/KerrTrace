mod input;
use std::{collections::HashSet, num::NonZeroU32, path::Path, sync::Arc, time::Instant};

use anyhow::{Context, Result, anyhow};
use glam::Vec3;
use softbuffer::{Context as SoftContext, Surface};
use winit::{keyboard::KeyCode, window::Window};

use crate::{
    Config,
    hud::{self, HudLayout, TextStyle, draw_hud},
    math::{calculate_camera_basis, ensure_finite_f32, ensure_finite_vec3},
    renderer::CudaRenderer,
};
pub struct App {
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
    has_frame: bool,
    fps_last_instant: Instant,
    fps_frames: u32,
    fps_value: f32,
}
impl App {
    const FOV_EPSILON: f32 = 1e-4;
    const ROTATION_EPSILON: f32 = 1e-4;

    pub(crate) fn new(config: Config) -> Self {
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
            has_frame: false,
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

    fn update_camera(&mut self) -> Result<()> {
        let (fwd, rgt, _up) = calculate_camera_basis(self.cam_yaw, self.cam_pitch);
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
        self.clamp_camera_to_escape_radius()?;
        Ok(())
    }

    fn clamp_camera_to_escape_radius(&mut self) -> Result<()> {
        let cam_pos = ensure_finite_vec3(self.cam_pos, "摄像机位置")?;
        let escape_radius =
            ensure_finite_f32(self.config.kernel.integrator.escape_radius, "escape_radius")?;
        if escape_radius <= 0.0 {
            return Err(anyhow!("escape_radius 无效: {escape_radius}"));
        }
        let dist = ensure_finite_f32(cam_pos.length(), "摄像机距离")?;
        if dist > escape_radius {
            let scale = escape_radius / dist;
            self.cam_pos = cam_pos * scale;
        }
        Ok(())
    }

    fn render(&mut self) -> Result<()> {
        if self.window.is_none() {
            return Ok(());
        }
        if self.renderer.is_none() || self.surface.is_none() {
            return Ok(());
        }
        let rendered = self.update_render_if_needed()?;
        self.update_fps(rendered);
        self.present_frame()?;
        self.throttle_if_needed();
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
        Ok(())
    }

    fn update_render_if_needed(&mut self) -> Result<bool> {
        if !self.should_render() {
            return Ok(false);
        }
        let (fwd, rgt, up) = calculate_camera_basis(self.cam_yaw, self.cam_pitch);
        let fov_scale = (self.fov.to_radians() / 2.0).tan();
        let save_first_frame = self.config.renderer.save_first_frame;
        let first_frame_path = self.config.renderer.first_frame_path.clone();
        let width = self.config.window.width;
        let height = self.config.window.height;
        {
            let renderer = self.renderer.as_mut().context("Renderer not initialized")?;
            let buffer_u32 = renderer.render(
                self.cam_pos.to_array(),
                fwd.to_array(),
                rgt.to_array(),
                up.to_array(),
                fov_scale,
            )?;
            if save_first_frame {
                let path = Path::new(&first_frame_path);
                if !path.exists() {
                    let mut buffer_u8 = Vec::with_capacity(buffer_u32.len() * 4);
                    for &pixel in buffer_u32 {
                        let r = ((pixel >> 16) & 0xFF) as u8;
                        let g = ((pixel >> 8) & 0xFF) as u8;
                        let b = (pixel & 0xFF) as u8;
                        buffer_u8.extend_from_slice(&[r, g, b, 255]);
                    }
                    image::save_buffer(path, &buffer_u8, width, height, image::ColorType::Rgba8)?;
                }
            }
        }
        self.has_frame = true;
        self.prev_cam_pos = self.cam_pos;
        self.prev_cam_yaw = self.cam_yaw;
        self.prev_cam_pitch = self.cam_pitch;
        self.prev_fov = self.fov;
        Ok(true)
    }

    fn should_render(&self) -> bool {
        if !self.has_frame {
            return true;
        }
        let position_delta = (self.cam_pos - self.prev_cam_pos).length();
        if position_delta > 0.0 {
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
        if !self.has_frame {
            return Err(anyhow!("Missing render buffer"));
        }
        let renderer = self.renderer.as_ref().context("Renderer not initialized")?;
        let buffer_u32 = renderer.host_image()?;
        let width = self.config.window.width;
        let height = self.config.window.height;
        let hud_layout = self.build_hud_layout(width, height)?;
        let surface = self.surface.as_mut().context("Surface not initialized")?;
        let mut buffer = surface
            .buffer_mut()
            .map_err(|e| anyhow::anyhow!("Failed to access surface buffer: {e}"))?;
        if buffer.len() != buffer_u32.len() {
            return Err(anyhow!(
                "Surface buffer size mismatch: surface={} frame={}",
                buffer.len(),
                buffer_u32.len()
            ));
        }
        buffer.copy_from_slice(buffer_u32);
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
}
