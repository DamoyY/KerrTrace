use super::{
    defines::build_cuda_defines, disk::build_kerr_params, pipeline::KerrParams,
    texture::CudaTextureLut,
};
use crate::config;
use anyhow::{Context as _, Result, ensure};
use core::mem;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg as _},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use std::path::Path;
#[test]
#[ignore = "需要 NVIDIA GPU 与 CUDA Toolkit"]
fn device_formulas_obey_metric_and_transfer() -> Result<()> {
    let mut settings = config::parse(include_str!("../../assets/settings.yaml"))?;
    settings.disk_noise.enabled = false;
    settings.kernel.disk.temperature_scale = 1000.0_f32;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let colors: Vec<f32> = [0.0_f32, 0.25_f32, 0.5_f32, 0.75_f32, 1.0_f32]
        .into_iter()
        .flat_map(|value| [value, value, value, 0.0_f32])
        .collect();
    let spectrum = CudaTextureLut::new(&context, &colors, 5, 4)?;
    let disk = CudaTextureLut::new(
        &context,
        &[0.0_f32, 0.25_f32, 0.5_f32, 0.75_f32, 1.0_f32],
        5,
        1,
    )?;
    for spin in [-0.45_f32, 0.0_f32, 0.45_f32] {
        settings.kernel.black_hole.spin = spin;
        let params = build_kerr_params(&settings)?;
        for fast_math in [false, true] {
            let source = format!(
                "{}\n{}\n{}\n{}",
                build_cuda_defines(&settings.kernel),
                include_str!("../../assets/shaders/kernel.cu"),
                include_str!("probes/hamiltonian.cuh"),
                include_str!("probes/transfer.cuh"),
            );
            let mut options = vec![String::from("--warning-as-error=all-warnings")];
            if fast_math {
                options.push(String::from("--use_fast_math"));
            }
            let ptx = compile_ptx_with_opts(
                &source,
                CompileOptions {
                    include_paths: vec![
                        Path::new("assets/shaders")
                            .to_str()
                            .context("路径不是 UTF-8")?
                            .into(),
                    ],
                    options,
                    ..Default::default()
                },
            )?;
            let module = context.load_module(ptx)?;
            let mut symbol = module.get_global("c_params", &stream)?;
            ensure!(
                symbol.len() == mem::size_of::<KerrParams>(),
                "参数 ABI 不匹配"
            );
            let mut view =
                unsafe { symbol.transmute_mut::<KerrParams>(1) }.context("参数对齐异常")?;
            stream.memcpy_htod(core::slice::from_ref(&params), &mut view)?;
            let mut errors = stream.alloc_zeros::<u32>(2)?;
            let mut output = stream.alloc_zeros::<f32>(8)?;
            let probe = module.load_function("formula_probe")?;
            unsafe {
                stream
                    .launch_builder(&probe)
                    .arg(&mut output)
                    .arg(&mut errors)
                    .arg(&spectrum.texture)
                    .arg(&disk.texture)
                    .launch(LaunchConfig::for_num_elems(1))
            }?;
            let values = stream.clone_dtoh(&output)?;
            let flags = stream.clone_dtoh(&errors)?;
            ensure!(flags.first() == Some(&0), "探针报告设备错误: {flags:?}");
            let &[
                gradient_error,
                color,
                transmission,
                time_direction,
                null_constraint,
                orbit_norm,
                zero_error_scale,
                nonfinite_error_scale,
            ] = values.as_slice()
            else {
                anyhow::bail!("探针输出尺寸错误");
            };
            let b = spin.abs() * (params.m / 5.0_f32.powi(3)).sqrt();
            let redshift = 2.0_f32
                .mul_add(b, 1.0_f32 - 3.0_f32 * params.m / 5.0_f32)
                .sqrt()
                / (1.0_f32 + b);
            let direction = if spin < 0.0_f32 { -1.0_f32 } else { 1.0_f32 };
            let omega = direction * (params.m / 5.0_f32.powi(3)).sqrt() / (1.0_f32 + b);
            let angular_momentum = 0.1_f32 * (25.0_f32 + params.aa).sqrt();
            let intensity = 0.25_f32 * redshift / omega.mul_add(angular_momentum, 1.0_f32);
            let expected_transmission = (-0.5_f32 * intensity).exp();
            ensure!(
                time_direction < 0.0_f32,
                "相机光线不是过去指向: k^t={time_direction}"
            );
            ensure!(
                null_constraint.abs() < 2e-6_f32,
                "相机光线未满足零测地线条件: {null_constraint}"
            );
            ensure!(
                (orbit_norm + 1.0_f32).abs() < 2e-6_f32,
                "盘四速度未归一化: {orbit_norm}"
            );
            ensure!(
                (zero_error_scale - 5.0_f32).abs() < f32::EPSILON
                    && (nonfinite_error_scale - 0.2_f32).abs() < f32::EPSILON,
                "步长更新未正确处理边界值"
            );
            ensure!(
                gradient_error < 2e-5_f32,
                "哈密顿偏导不匹配: {gradient_error}, spin={spin}, fast_math={fast_math}"
            );
            ensure!(
                (transmission - expected_transmission).abs() < 3e-4_f32
                    && (-intensity)
                        .mul_add(1.0_f32 - expected_transmission, color)
                        .abs()
                        < 3e-4_f32,
                "盘红移/纹理/不透明度不匹配: color={color}, transmission={transmission}, expected={expected_transmission}, spin={spin}"
            );
        }
    }
    Ok(())
}
