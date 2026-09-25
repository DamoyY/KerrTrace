use crate::{config, config::numbers::calculate_camera_basis, renderer::CudaRenderer};
use anyhow::{Result, ensure};
use core::time::Duration;
use std::{path::Path, time::Instant};
#[test]
#[ignore = "需要 NVIDIA GPU 与 CUDA Toolkit"]
fn cuda_pipeline_compiles_and_renders() -> Result<()> {
    let mut settings = config::parse(include_str!("../../assets/settings.yaml"))?;
    settings.window.width = 129;
    settings.window.height = 67;
    settings.kernel.spp = 1;
    for bloom in [false, true] {
        settings.bloom.enabled = bloom;
        settings.cuda.use_fast_math = bloom;
        let mut renderer = CudaRenderer::new(&settings, Path::new("assets/shaders"))?;
        ensure!(submit(&mut renderer, &settings)?, "首帧未提交");
        ensure!(submit(&mut renderer, &settings)?, "第二帧未提交");
        ensure!(!submit(&mut renderer, &settings)?, "帧队列未施加背压");
        consume(&mut renderer)?;
        consume(&mut renderer)?;
        ensure!(renderer.ready_frame()?.is_none(), "帧队列未清空");
        ensure!(renderer.displayed_frame()?.is_some(), "已显示的帧未保留");
        ensure!(submit(&mut renderer, &settings)?, "帧缓冲区未回收");
        consume(&mut renderer)?;
    }
    Ok(())
}
fn submit(renderer: &mut CudaRenderer, settings: &config::Config) -> Result<bool> {
    let (forward, right, up) = calculate_camera_basis(settings.camera.yaw, settings.camera.pitch)?;
    renderer.submit_render(
        settings.camera.position,
        forward.to_array(),
        right.to_array(),
        up.to_array(),
        (settings.camera.fov.to_radians() / 2.0_f32).tan(),
    )
}
fn consume(renderer: &mut CudaRenderer) -> Result<()> {
    let start = Instant::now();
    loop {
        if let Some(frame) = renderer.ready_frame()? {
            ensure!(frame.len() == 129 * 67, "帧尺寸不正确");
            ensure!(frame.iter().any(|&pixel| pixel != 0), "渲染结果全黑");
            renderer.finish_frame()?;
            return Ok(());
        }
        ensure!(start.elapsed() < Duration::from_secs(30), "GPU 帧等待超时");
        std::thread::sleep(Duration::from_millis(5));
    }
}
