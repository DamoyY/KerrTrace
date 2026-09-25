use super::Config;
use anyhow::{Context as _, Result, ensure};
pub(super) fn validate(config: &Config) -> Result<()> {
    let window = &config.window;
    let pixels = window
        .width
        .checked_mul(window.height)
        .context("窗口像素数溢出")?;
    ensure!(window.width > 0 && window.height > 0, "窗口尺寸应大于 0");
    ensure!(
        i32::try_from(pixels).is_ok(),
        "窗口像素数超出 CUDA i32 索引范围"
    );
    let [block_x, block_y] = config.renderer.block_dim;
    let threads = block_x
        .checked_mul(block_y)
        .context("CUDA 线程块尺寸溢出")?;
    ensure!(
        block_x > 0 && block_y > 0 && threads <= 1024,
        "CUDA 线程块应包含 1..=1024 个线程"
    );
    ensure!(
        config.hud.font_size > 0 && config.hud.font_size <= 256,
        "HUD 字号应为 1..=256"
    );
    ensure!(
        config.hud.margin.iter().all(|&value| value <= 0x8000),
        "HUD 边距过大"
    );
    ensure!(config.hud.color[3] == 255, "HUD 仅支持不透明颜色");
    ensure!(
        !config.renderer.first_frame_path.is_empty(),
        "首帧路径不能为空"
    );
    let camera = &config.camera;
    bounds(
        "camera.pitch_limit",
        camera.pitch_limit,
        -89.9_f32,
        89.9_f32,
    )?;
    bounds("camera.fov_limit", camera.fov_limit, 0.1_f32, 179.0_f32)?;
    ensure!(
        (camera.pitch_limit[0]..=camera.pitch_limit[1]).contains(&camera.pitch),
        "pitch 超出范围"
    );
    ensure!(
        (camera.fov_limit[0]..=camera.fov_limit[1]).contains(&camera.fov),
        "fov 超出范围"
    );
    for value in camera.position {
        finite("camera.position", value)?;
    }
    finite("camera.yaw", camera.yaw)?;
    positive("camera.zoom_speed", camera.zoom_speed)?;
    positive("controls.move_speed", config.controls.move_speed)?;
    positive(
        "controls.sprint_multiplier",
        config.controls.sprint_multiplier,
    )?;
    positive(
        "controls.mouse_sensitivity",
        config.controls.mouse_sensitivity,
    )?;
    nonnegative("bloom.intensity", config.bloom.intensity)?;
    nonnegative("bloom.radius", config.bloom.radius)?;
    ensure!(config.bloom.radius <= 4096.0_f32, "Bloom 半径过大");
    let blackbody = &config.blackbody;
    ensure!(
        (2..=0x0010_0000).contains(&blackbody.lut_size),
        "LUT 尺寸应为 2..=1048576"
    );
    positive("blackbody.lut_max_temp", blackbody.lut_max_temp)?;
    positive("blackbody.wavelength_start", blackbody.wavelength_start)?;
    positive("blackbody.wavelength_end", blackbody.wavelength_end)?;
    positive("blackbody.wavelength_step", blackbody.wavelength_step)?;
    ensure!(
        blackbody.wavelength_end >= blackbody.wavelength_start,
        "波长范围颠倒"
    );
    let samples = (f64::from(blackbody.wavelength_end) - f64::from(blackbody.wavelength_start))
        / f64::from(blackbody.wavelength_step);
    ensure!(samples <= 65535.0_f64, "波长采样数量过大");
    ensure!(
        blackbody.wavelength_start + blackbody.wavelength_step > blackbody.wavelength_start,
        "波长步长小于浮点精度"
    );
    validate_physics(config)
}
fn validate_physics(config: &Config) -> Result<()> {
    let kernel = &config.kernel;
    let hole = &kernel.black_hole;
    positive("black_hole.mass", hole.mass)?;
    finite("black_hole.spin", hole.spin)?;
    ensure!(
        hole.spin.abs() < hole.mass,
        "仅支持 |spin| < mass 的非极端 Kerr 黑洞"
    );
    let integrator = &kernel.integrator;
    for (label, value) in [
        ("exposure_scale", kernel.exposure_scale),
        ("sky.line_thickness", kernel.sky.line_thickness),
        ("disk.outer_radius", kernel.disk.outer_radius),
        ("disk.temperature_scale", kernel.disk.temperature_scale),
        ("integrator.initial_step", integrator.initial_step),
        ("integrator.tolerance", integrator.tolerance),
        ("integrator.horizon_epsilon", integrator.horizon_epsilon),
        ("integrator.escape_radius", integrator.escape_radius),
    ] {
        positive(label, value)?;
    }
    nonnegative("sky.intensity", kernel.sky.intensity)?;
    for (label, value) in [
        ("spp", kernel.spp),
        ("sky.grid_divisions", kernel.sky.grid_divisions),
        ("integrator.max_steps", integrator.max_steps),
        ("integrator.max_attempts", integrator.max_attempts),
    ] {
        ensure!(
            value > 0 && i32::try_from(value).is_ok(),
            "{label} 超出 1..=i32::MAX"
        );
    }
    ensure!(
        (0.0_f32..1.0_f32).contains(&integrator.transmittance_cutoff),
        "transmittance_cutoff 应在 [0, 1) 内"
    );
    ensure!(
        integrator.escape_radius >= kernel.disk.outer_radius,
        "逃逸半径小于吸积盘外半径"
    );
    let noise = &config.disk_noise;
    positive("disk_noise.scale", noise.scale)?;
    finite("disk_noise.winding", noise.winding)?;
    ensure!(
        (0.0_f32..=2.0_f32).contains(&noise.strength),
        "disk_noise.strength 应在 [0, 2] 内"
    );
    ensure!(
        (1..=24).contains(&noise.detail),
        "disk_noise.detail 应在 1..=24 内"
    );
    Ok(())
}
fn bounds(label: &str, values: [f32; 2], min: f32, max: f32) -> Result<()> {
    ensure!(
        values[0] >= min && values[1] <= max && values[0] <= values[1],
        "{label} 应是 {min}..={max} 内的有序区间"
    );
    Ok(())
}
fn finite(label: &str, value: f32) -> Result<()> {
    ensure!(value.is_finite(), "{label} 不是有限值: {value}");
    Ok(())
}
fn positive(label: &str, value: f32) -> Result<()> {
    finite(label, value)?;
    ensure!(value > 0.0_f32, "{label} 应大于 0");
    Ok(())
}
fn nonnegative(label: &str, value: f32) -> Result<()> {
    finite(label, value)?;
    ensure!(value >= 0.0_f32, "{label} 不能小于 0");
    Ok(())
}
