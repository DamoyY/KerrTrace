mod glyphs;
use anyhow::{Result, anyhow};
pub use glyphs::{GLYPH_HEIGHT, GLYPH_WIDTH};
use log::error;
#[derive(Clone, Copy)]
pub struct TextStyle {
    pub width: u32,
    pub height: u32,
    pub color: [u8; 4],
    pub scale: u32,
}
pub struct HudLayout {
    pub style: TextStyle,
    pub margin_x: i32,
    pub margin_y: i32,
    pub fps_y: i32,
    pub info_text: String,
    pub fps_text: String,
}
pub fn draw_hud(buffer: &mut [u32], layout: &HudLayout) {
    draw_text(
        buffer,
        &layout.style,
        layout.margin_x,
        layout.margin_y,
        &layout.info_text,
    );
    draw_text(
        buffer,
        &layout.style,
        layout.margin_x,
        layout.fps_y,
        &layout.fps_text,
    );
}
pub fn draw_text(buffer: &mut [u32], style: &TextStyle, x: i32, y: i32, text: &str) {
    let mut context = match DrawContext::new(buffer, style) {
        Ok(context) => context,
        Err(err) => {
            error!("HUD 初始化失败: {err}");
            return;
        }
    };
    if let Err(err) = context.draw_text(x, y, text) {
        error!("HUD 绘制失败: {err}");
    }
}
struct DrawContext<'a> {
    buffer: &'a mut [u32],
    width: i32,
    height: i32,
    width_usize: usize,
    color: u32,
    scale: i32,
}
impl<'a> DrawContext<'a> {
    fn new(buffer: &'a mut [u32], style: &TextStyle) -> Result<Self> {
        if style.scale == 0 {
            return Err(anyhow!("HUD 字体缩放比例不能为 0"));
        }
        let width = i32::try_from(style.width)
            .map_err(|_| anyhow!("HUD 宽度超出 i32 范围: {}", style.width))?;
        let height = i32::try_from(style.height)
            .map_err(|_| anyhow!("HUD 高度超出 i32 范围: {}", style.height))?;
        let width_usize =
            usize::try_from(width).map_err(|_| anyhow!("HUD 宽度无法转换为 usize: {width}"))?;
        let scale = i32::try_from(style.scale)
            .map_err(|_| anyhow!("HUD 缩放比例超出 i32 范围: {}", style.scale))?;
        let color = (u32::from(style.color[0]) << 16)
            | (u32::from(style.color[1]) << 8)
            | u32::from(style.color[2]);
        Ok(Self {
            buffer,
            width,
            height,
            width_usize,
            color,
            scale,
        })
    }

    fn draw_text(&mut self, x: i32, y: i32, text: &str) -> Result<()> {
        let mut cursor_x = x;
        let mut cursor_y = y;
        let line_height = (GLYPH_HEIGHT + 1)
            .checked_mul(self.scale)
            .ok_or_else(|| anyhow!("HUD 行高计算溢出"))?;
        let glyph_advance = (GLYPH_WIDTH + 1)
            .checked_mul(self.scale)
            .ok_or_else(|| anyhow!("HUD 字符步进计算溢出"))?;
        for ch in text.chars() {
            if ch == '\n' {
                cursor_x = x;
                cursor_y += line_height;
                continue;
            }
            self.draw_char(cursor_x, cursor_y, ch)?;
            cursor_x += glyph_advance;
        }
        Ok(())
    }

    fn draw_char(&mut self, x: i32, y: i32, ch: char) -> Result<()> {
        let glyph = glyphs::glyph_pattern(ch);
        for (row, bits) in glyph.iter().enumerate() {
            let row_i =
                i32::try_from(row).map_err(|_| anyhow!("HUD 字符行索引超出 i32 范围: {row}"))?;
            for col in 0..GLYPH_WIDTH {
                let shift = u32::try_from(GLYPH_WIDTH - 1 - col)
                    .map_err(|_| anyhow!("HUD 位移量超出 u32 范围: {}", GLYPH_WIDTH - 1 - col))?;
                let mask = 1u8 << shift;
                if (bits & mask) == 0 {
                    continue;
                }
                let px = x + col * self.scale;
                let py = y + row_i * self.scale;
                for dy in 0..self.scale {
                    let iy = py + dy;
                    if iy < 0 || iy >= self.height {
                        continue;
                    }
                    let row_index =
                        usize::try_from(iy).map_err(|_| anyhow!("HUD 行索引无法转换: {iy}"))?;
                    let row_offset = row_index
                        .checked_mul(self.width_usize)
                        .ok_or_else(|| anyhow!("HUD 行偏移计算溢出"))?;
                    for dx in 0..self.scale {
                        let ix = px + dx;
                        if ix < 0 || ix >= self.width {
                            continue;
                        }
                        let col_index =
                            usize::try_from(ix).map_err(|_| anyhow!("HUD 列索引无法转换: {ix}"))?;
                        let index = row_offset
                            .checked_add(col_index)
                            .ok_or_else(|| anyhow!("HUD 像素索引计算溢出"))?;
                        let pixel = self
                            .buffer
                            .get_mut(index)
                            .ok_or_else(|| anyhow!("HUD 像素索引越界: {index}"))?;
                        *pixel = self.color;
                    }
                }
            }
        }
        Ok(())
    }
}
