use crate::{config, config::numbers::f32_from_f64};
use anyhow::Result;
const SETTINGS: &str = include_str!("../../assets/settings.yaml");
#[test]
fn default_configuration_is_valid() -> Result<()> {
    config::parse(SETTINGS)?;
    Ok(())
}
#[test]
fn unsafe_configuration_is_rejected() {
    for (original, invalid) in [
        ("width: 2100", "width: 0"),
        ("block_dim: [16, 16]", "block_dim: [1024, 1024]"),
        ("mass: 0.5", "mass: -0.5"),
        ("spin: -0.45", "spin: 0.6"),
        ("fov_limit: [1.0, 120.0]", "fov_limit: [120.0, 1.0]"),
        ("pitch_limit: [-80.0, 80.0]", "pitch_limit: [-90.0, 90.0]"),
        ("spp: 4", "spp: 0"),
        ("max_steps: 100", "max_steps: 0"),
        ("tolerance: 1e-5", "tolerance: .nan"),
        ("wavelength_step: 10.0", "wavelength_step: 1e-30"),
        ("lut_size: 4096", "lut_size: 1"),
        ("height: 900", "height: 2147483647"),
    ] {
        assert!(
            SETTINGS.contains(original),
            "fixture field missing: {original}"
        );
        let invalid_source = SETTINGS.replace(original, invalid);
        assert!(
            config::parse(&invalid_source).is_err(),
            "accepted unsafe configuration: {invalid}"
        );
    }
}
#[test]
fn unknown_configuration_fields_are_rejected() {
    let invalid = format!("{SETTINGS}\nunknown_option: true\n");
    config::parse(&invalid).unwrap_err();
}
#[test]
fn float_conversion_rejects_nonfinite_and_overflow() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        f32_from_f64(value).unwrap_err();
    }
    assert!(f32_from_f64(f64::from(f32::MAX)).unwrap().is_finite());
    assert!((f32_from_f64(1.25_f64).unwrap() - 1.25_f32).abs() < f32::EPSILON);
}
