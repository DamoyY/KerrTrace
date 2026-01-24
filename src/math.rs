use anyhow::{Result, anyhow};
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
pub fn f32_from_f64(value: f64) -> Result<f32> {
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
pub fn f32_from_f64_with_context(value: f64, label: &str) -> Result<f32> {
    f32_from_f64(value).map_err(|err| anyhow!("{label}转换失败: {err}"))
}
pub fn ensure_finite_f32(value: f32, label: &str) -> Result<f32> {
    if !value.is_finite() {
        return Err(anyhow!("{label}不是有限值: {value}"));
    }
    Ok(value)
}
pub fn ensure_finite_vec3(value: Vec3, label: &str) -> Result<Vec3> {
    if !value.is_finite() {
        return Err(anyhow!("{label}不是有限值: {value:?}"));
    }
    Ok(value)
}
