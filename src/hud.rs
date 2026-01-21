use anyhow::{Result, anyhow};
use log::error;

pub const GLYPH_WIDTH: i32 = 5;
pub const GLYPH_HEIGHT: i32 = 7;

#[derive(Clone, Copy)]
pub struct TextStyle {
    pub width: u32,
    pub height: u32,
    pub color: [u8; 4],
    pub scale: u32,
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
        let glyph = glyph_pattern(ch);
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

const GLYPH_TABLE: &[(char, [u8; 7])] = &[
    (
        '0',
        [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
    ),
    (
        '1',
        [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        '2',
        [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
    ),
    (
        '3',
        [
            0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
        ],
    ),
    (
        '4',
        [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
    ),
    (
        '5',
        [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
    ),
    (
        '6',
        [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        '7',
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
    ),
    (
        '8',
        [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        '9',
        [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b11100,
        ],
    ),
    (
        'A',
        [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'B',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        'C',
        [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
    ),
    (
        'D',
        [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
    ),
    (
        'E',
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        'F',
        [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        'G',
        [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        'H',
        [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'I',
        [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
    ),
    (
        'J',
        [
            0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        'K',
        [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        'L',
        [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
    ),
    (
        'M',
        [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'N',
        [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
    ),
    (
        'O',
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        'P',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
    ),
    (
        'Q',
        [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
    ),
    (
        'R',
        [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
    ),
    (
        'S',
        [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
    ),
    (
        'T',
        [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        'U',
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
    ),
    (
        'V',
        [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
    ),
    (
        'W',
        [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
    ),
    (
        'X',
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
    ),
    (
        'Y',
        [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
    ),
    (
        'Z',
        [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
    ),
    (
        ':',
        [
            0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000,
        ],
    ),
    (
        '.',
        [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100,
        ],
    ),
    (
        '-',
        [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
    ),
    (
        '=',
        [
            0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000,
        ],
    ),
];

fn glyph_pattern(ch: char) -> [u8; 7] {
    let upper = ch.to_ascii_uppercase();
    for (key, pattern) in GLYPH_TABLE {
        if *key == upper {
            return *pattern;
        }
    }
    [0b00000; 7]
}
