mod camera;
mod events;
mod input;
mod render;
use crate::config::Config;
use alloc::sync::Arc;
use camera::Camera;
use hashbrown::HashSet;
use std::time::Instant;
use winit::{keyboard::KeyCode, window::Window};
pub struct App {
    pub(super) config: Config,
    pub(super) window: Option<Arc<Window>>,
    pub(super) surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    pub(super) renderer: Option<crate::renderer::CudaRenderer>,
    pub(super) context: Option<softbuffer::Context<Arc<Window>>>,
    camera: Camera,
    last_rendered: Option<Camera>,
    pub(super) keys_pressed: HashSet<KeyCode>,
    pub(super) mouse_locked: bool,
    pub(super) last_present: Instant,
    pub(super) last_camera_update: Instant,
    pub(super) save_first_frame_pending: bool,
    pub(super) fps_last_instant: Instant,
    pub(super) fps_frames: u32,
    pub(super) fps_value: f32,
    pub(crate) failure: Option<anyhow::Error>,
}
impl App {
    pub(crate) fn new(config: Config) -> Self {
        let camera = Camera {
            position: glam::Vec3::from_array(config.camera.position),
            yaw: config.camera.yaw,
            pitch: config.camera.pitch,
            fov: config.camera.fov,
        };
        let now = Instant::now();
        Self {
            camera,
            last_rendered: None,
            save_first_frame_pending: config.renderer.save_first_frame,
            config,
            window: None,
            surface: None,
            renderer: None,
            context: None,
            keys_pressed: HashSet::new(),
            mouse_locked: false,
            last_present: now,
            last_camera_update: now,
            fps_last_instant: now,
            fps_frames: 0,
            fps_value: 0.0_f32,
            failure: None,
        }
    }
}
