use super::App;
use crate::config::numbers::{
    calculate_camera_basis, ensure_finite_f32, ensure_finite_vec3, f32_from_f64,
};
use anyhow::{Result, anyhow};
use glam::Vec3;
use winit::keyboard::KeyCode;
#[derive(Clone, Copy, PartialEq)]
pub(super) struct Camera {
    pub(super) position: Vec3,
    pub(super) yaw: f32,
    pub(super) pitch: f32,
    pub(super) fov: f32,
}
impl App {
    pub(super) fn update_camera(&mut self) -> Result<()> {
        let (forward, right, _) = calculate_camera_basis(self.camera.yaw, self.camera.pitch)?;
        let elapsed = self.last_camera_update.elapsed();
        self.last_camera_update = std::time::Instant::now();
        let elapsed_seconds = f32_from_f64(elapsed.as_secs_f64())?;
        let mut speed = self.config.controls.move_speed * elapsed_seconds;
        if self.keys_pressed.contains(&KeyCode::ShiftLeft) {
            speed *= self.config.controls.sprint_multiplier;
        }
        if self.keys_pressed.contains(&KeyCode::KeyW) {
            self.camera.position += forward * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) {
            self.camera.position -= forward * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyA) {
            self.camera.position -= right * speed;
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) {
            self.camera.position += right * speed;
        }
        if self.keys_pressed.contains(&KeyCode::Space) {
            self.camera.position.y += speed;
        }
        if self.keys_pressed.contains(&KeyCode::ControlLeft) {
            self.camera.position.y -= speed;
        }
        self.clamp_camera_to_escape_radius()
    }
    fn clamp_camera_to_escape_radius(&mut self) -> Result<()> {
        let camera_position = ensure_finite_vec3(self.camera.position, "摄像机位置")?;
        let escape_radius =
            ensure_finite_f32(self.config.kernel.integrator.escape_radius, "escape_radius")?;
        if escape_radius <= 0.0_f32 {
            return Err(anyhow!("escape_radius 无效: {escape_radius}"));
        }
        let distance = ensure_finite_f32(camera_position.length(), "摄像机距离")?;
        if distance > escape_radius {
            self.camera.position = camera_position * (escape_radius / distance);
        }
        Ok(())
    }
}
