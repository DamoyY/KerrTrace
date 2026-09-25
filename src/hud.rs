use anyhow::{Context as _, Result, anyhow};
use font8x8::UnicodeFonts as _;
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
pub fn draw_hud(buffer: &mut [u32], layout: &HudLayout) -> Result<()> {
    let mut context = DrawContext::new(buffer, layout.style)?;
    context.draw_text(layout.margin_x, layout.margin_y, &layout.info_text)?;
    context.draw_text(layout.margin_x, layout.fps_y, &layout.fps_text)?;
    Ok(())
}
struct DrawContext<'buffer> {
    buffer: &'buffer mut [u32],
    width: i32,
    height: i32,
    width_usize: usize,
    color: u32,
    scale: i32,
}
impl<'buffer> DrawContext<'buffer> {
    fn new(buffer: &'buffer mut [u32], style: TextStyle) -> Result<Self> {
        if style.scale == 0 {
            return Err(anyhow!("HUD 字体缩放比例不能为 0"));
        }
        let width = i32::try_from(style.width).context("HUD 宽度超出 i32 范围")?;
        let height = i32::try_from(style.height).context("HUD 高度超出 i32 范围")?;
        let width_usize = usize::try_from(width).context("HUD 宽度无法转换为 usize")?;
        let expected_len = width_usize
            .checked_mul(usize::try_from(height).context("HUD 高度无法转换为 usize")?)
            .context("HUD 缓冲区尺寸溢出")?;
        if buffer.len() != expected_len {
            return Err(anyhow!(
                "HUD 缓冲区大小不匹配: expected={expected_len} actual={}",
                buffer.len()
            ));
        }
        let scale = i32::try_from(style.scale).context("HUD 缩放比例超出 i32 范围")?;
        let color = (u32::from(style.color[0]) << 16_u32)
            | (u32::from(style.color[1]) << 8_u32)
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
        let line_height = 9_i32.checked_mul(self.scale).context("HUD 行高计算溢出")?;
        let glyph_advance = 9_i32
            .checked_mul(self.scale)
            .context("HUD 字符步进计算溢出")?;
        let mut cursor_x = x;
        let mut cursor_y = y;
        for character in text.chars() {
            if character == '\n' {
                cursor_x = x;
                cursor_y = cursor_y
                    .checked_add(line_height)
                    .context("HUD 垂直位置计算溢出")?;
                continue;
            }
            self.draw_character(cursor_x, cursor_y, character)?;
            cursor_x = cursor_x
                .checked_add(glyph_advance)
                .context("HUD 水平位置计算溢出")?;
        }
        Ok(())
    }
    fn draw_character(&mut self, x: i32, y: i32, character: char) -> Result<()> {
        let glyph = font8x8::BASIC_FONTS
            .get(character)
            .ok_or_else(|| anyhow!("HUD 不支持字符 {character:?}"))?;
        for (row_index, bits) in glyph.iter().enumerate() {
            let row = i32::try_from(row_index).context("HUD 字符行索引超出 i32 范围")?;
            for bit in 0_u32..8_u32 {
                if bits & (1_u8 << bit) == 0 {
                    continue;
                }
                let pixel_x = x
                    .checked_add(
                        i32::try_from(bit)
                            .context("HUD 字符列索引超出 i32 范围")?
                            .checked_mul(self.scale)
                            .context("HUD 字符列位置计算溢出")?,
                    )
                    .context("HUD 字符水平位置计算溢出")?;
                let pixel_y = y
                    .checked_add(
                        row.checked_mul(self.scale)
                            .context("HUD 字符行位置计算溢出")?,
                    )
                    .context("HUD 字符垂直位置计算溢出")?;
                for offset_y in 0_i32..self.scale {
                    for offset_x in 0_i32..self.scale {
                        let target_x = pixel_x.checked_add(offset_x).context("HUD X 坐标溢出")?;
                        let target_y = pixel_y.checked_add(offset_y).context("HUD Y 坐标溢出")?;
                        self.set_pixel(target_x, target_y)?;
                    }
                }
            }
        }
        Ok(())
    }
    fn set_pixel(&mut self, x: i32, y: i32) -> Result<()> {
        if x < 0_i32 || y < 0_i32 || x >= self.width || y >= self.height {
            return Ok(());
        }
        let row = usize::try_from(y).context("HUD 行索引无法转换")?;
        let column = usize::try_from(x).context("HUD 列索引无法转换")?;
        let index = row
            .checked_mul(self.width_usize)
            .context("HUD 行偏移溢出")?
            .checked_add(column)
            .context("HUD 像素索引计算溢出")?;
        let pixel = self.buffer.get_mut(index).context("HUD 像素索引越界")?;
        *pixel = self.color;
        Ok(())
    }
}
