use crate::hud::{HudLayout, TextStyle, draw_hud};
use anyhow::Result;
fn layout() -> HudLayout {
    HudLayout {
        style: TextStyle {
            width: 20,
            height: 12,
            color: [255, 255, 255, 255],
            scale: 1,
        },
        margin_x: -3,
        margin_y: -2,
        fps_y: 8,
        info_text: String::from("POS: 1 2 3\nFOV: 15"),
        fps_text: String::from("FPS: 60"),
    }
}
#[test]
fn hud_clips_to_frame_edges() -> Result<()> {
    let mut pixels = vec![0_u32; 240];
    draw_hud(&mut pixels, &layout())?;
    anyhow::ensure!(pixels.iter().any(|&pixel| pixel != 0), "HUD 没有绘制像素");
    Ok(())
}
#[test]
fn hud_rejects_invalid_storage_and_coordinates() {
    let mut pixels = vec![0_u32; 239];
    draw_hud(&mut pixels, &layout()).unwrap_err();
    pixels.push(0);
    let mut invalid = layout();
    invalid.margin_x = i32::MAX;
    draw_hud(&mut pixels, &invalid).unwrap_err();
    invalid.margin_x = 0_i32;
    invalid.info_text = String::from("不支持的字符");
    draw_hud(&mut pixels, &invalid).unwrap_err();
}
