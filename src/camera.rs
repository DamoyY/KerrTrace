use glam::Vec3;
pub fn calculate_camera_basis(yaw: f32, pitch: f32) -> (Vec3, Vec3, Vec3) {
    let yaw_rad = yaw.to_radians();
    let pitch_rad = pitch.to_radians();
    let fx = yaw_rad.sin() * pitch_rad.cos();
    let fy = pitch_rad.sin();
    let fz = -yaw_rad.cos() * pitch_rad.cos();
    let forward = Vec3::new(fx, fy, fz).normalize();
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = forward.cross(world_up).normalize();
    let up = right.cross(forward).normalize();
    (forward, right, up)
}
