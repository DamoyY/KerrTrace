use anyhow::{Result, anyhow};
use glam::Vec3;
use num_traits::ToPrimitive as _;
pub fn calculate_camera_basis(yaw: f32, pitch: f32) -> Result<(Vec3, Vec3, Vec3)> {
    let yaw_rad = yaw.to_radians();
    let pitch_rad = pitch.to_radians();
    let fx = yaw_rad.sin() * pitch_rad.cos();
    let fy = pitch_rad.sin();
    let fz = -yaw_rad.cos() * pitch_rad.cos();
    let forward = Vec3::new(fx, fy, fz)
        .try_normalize()
        .ok_or_else(|| anyhow!("摄像机朝向无法归一化"))?;
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let right = forward
        .cross(world_up)
        .try_normalize()
        .ok_or_else(|| anyhow!("摄像机右方向无法归一化"))?;
    let up = right
        .cross(forward)
        .try_normalize()
        .ok_or_else(|| anyhow!("摄像机上方向无法归一化"))?;
    Ok((forward, right, up))
}
pub fn f32_from_f64(value: f64) -> Result<f32> {
    if !value.is_finite() {
        return Err(anyhow!("数值不是有限值: {value}"));
    }
    let converted = value
        .to_f32()
        .ok_or_else(|| anyhow!("数值超出 f32 范围: {value}"))?;
    ensure_finite_f32(converted, "转换后的数值")
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
