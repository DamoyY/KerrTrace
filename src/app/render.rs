use super::App;
use crate::{
    config::numbers::calculate_camera_basis,
    hud::{HudLayout, TextStyle, draw_hud},
    renderer::CudaRenderer,
};
use alloc::sync::Arc;
use anyhow::{Context as _, Result, anyhow};
use core::num::NonZeroU32;
use image::ColorType;
use softbuffer::{Context as SoftContext, Surface};
use std::path::Path;
impl App {
    pub(super) fn init_renderer(&mut self) -> Result<()> {
        let window = self.window.as_ref().context("窗口尚未初始化")?;
        let context = SoftContext::new(Arc::clone(window))
            .map_err(|error| anyhow!("创建 softbuffer 上下文失败: {error}"))?;
        let mut surface = Surface::new(&context, Arc::clone(window))
            .map_err(|error| anyhow!("创建 softbuffer surface 失败: {error}"))?;
        let size = window.inner_size();
        self.config.window.width = size.width;
        self.config.window.height = size.height;
        let width = NonZeroU32::new(size.width).context("窗口宽度不能为 0")?;
        let height = NonZeroU32::new(size.height).context("窗口高度不能为 0")?;
        surface
            .resize(width, height)
            .map_err(|error| anyhow!("调整 softbuffer surface 大小失败: {error}"))?;
        let cuda_dir = std::env::current_dir()?.join("assets/shaders");
        let renderer = CudaRenderer::new(&self.config, &cuda_dir)?;
        self.context = Some(context);
        self.surface = Some(surface);
        self.renderer = Some(renderer);
        Ok(())
    }
    pub(super) fn render(&mut self) -> Result<()> {
        if self.window.is_none() || self.renderer.is_none() || self.surface.is_none() {
            return Err(anyhow!("渲染资源尚未初始化"));
        }
        self.update_render_if_needed()?;
        let presented = self.present_frame()?;
        self.update_fps(presented)?;
        self.last_present = std::time::Instant::now();
        Ok(())
    }
    fn update_render_if_needed(&mut self) -> Result<bool> {
        if self.last_rendered == Some(self.camera) {
            return Ok(false);
        }
        let (forward, right, up) = calculate_camera_basis(self.camera.yaw, self.camera.pitch)?;
        let fov_scale = (self.camera.fov.to_radians() / 2.0_f32).tan();
        let renderer = self.renderer.as_mut().context("渲染器尚未初始化")?;
        let submitted = renderer.submit_render(
            self.camera.position.to_array(),
            forward.to_array(),
            right.to_array(),
            up.to_array(),
            fov_scale,
        )?;
        if submitted {
            self.last_rendered = Some(self.camera);
        }
        Ok(submitted)
    }
    fn present_frame(&mut self) -> Result<bool> {
        let width = self.config.window.width;
        let height = self.config.window.height;
        let hud_layout = self.build_hud_layout(width, height)?;
        let save_first_frame = self.save_first_frame_pending;
        let first_frame_path = self.config.renderer.first_frame_path.clone();
        let mut clear_first_frame_flag = false;
        let renderer = self.renderer.as_mut().context("渲染器尚未初始化")?;
        let surface = self.surface.as_mut().context("surface 尚未初始化")?;
        let (ready_frame, fresh) = if let Some(frame) = renderer.ready_frame()? {
            (frame, true)
        } else if let Some(frame) = renderer.displayed_frame()? {
            (frame, false)
        } else {
            return Ok(false);
        };
        if save_first_frame {
            save_frame(&first_frame_path, ready_frame, width, height)?;
            clear_first_frame_flag = true;
        }
        let mut buffer = surface
            .buffer_mut()
            .map_err(|error| anyhow!("访问 surface 缓冲区失败: {error}"))?;
        if buffer.len() != ready_frame.len() {
            return Err(anyhow!(
                "surface 缓冲区大小不匹配: surface={} frame={}",
                buffer.len(),
                ready_frame.len()
            ));
        }
        buffer.copy_from_slice(ready_frame);
        draw_hud(&mut buffer, &hud_layout)?;
        buffer
            .present()
            .map_err(|error| anyhow!("呈现帧失败: {error}"))?;
        if fresh {
            renderer.finish_frame()?;
        }
        if clear_first_frame_flag {
            self.save_first_frame_pending = false;
        }
        Ok(fresh)
    }
    fn build_hud_layout(&self, width: u32, height: u32) -> Result<HudLayout> {
        let margin_x =
            i32::try_from(self.config.hud.margin[0]).context("HUD 横向边距超出 i32 范围")?;
        let margin_y =
            i32::try_from(self.config.hud.margin[1]).context("HUD 纵向边距超出 i32 范围")?;
        let font_size = self.config.hud.font_size;
        let scale = font_size.div_ceil(8_u32);
        let scale_i32 = i32::try_from(scale).context("HUD 缩放比例超出 i32 范围")?;
        let height_i32 = i32::try_from(height).context("窗口高度超出 i32 范围")?;
        let line_height = 9_i32.checked_mul(scale_i32).context("HUD 行高计算溢出")?;
        let fps_y = height_i32
            .checked_sub(margin_y)
            .context("HUD FPS 边距计算溢出")?
            .checked_sub(line_height)
            .context("HUD FPS 位置计算溢出")?;
        let style = TextStyle {
            width,
            height,
            color: self.config.hud.color,
            scale,
        };
        let info_text = format!(
            "POS : {:.1} {:.1} {:.1}\nVIEW: Y={:.1} P={:.1} FOV={:.0}",
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
            self.camera.yaw,
            self.camera.pitch,
            self.camera.fov
        );
        let fps_text = format!("FPS: {:.1}", self.fps_value);
        Ok(HudLayout {
            style,
            margin_x,
            margin_y,
            fps_y,
            info_text,
            fps_text,
        })
    }
}
fn save_frame(filename: &str, pixels: &[u32], width: u32, height: u32) -> Result<()> {
    let path = Path::new(filename);
    if path.try_exists()? {
        log::info!("首帧文件已存在，保留原文件: {}", path.display());
        return Ok(());
    }
    let capacity = pixels.len().checked_mul(4).context("首帧缓冲区尺寸溢出")?;
    let mut rgba = Vec::with_capacity(capacity);
    for &pixel in pixels {
        let [_, red, green, blue] = pixel.to_be_bytes();
        rgba.extend_from_slice(&[red, green, blue, 255_u8]);
    }
    image::save_buffer(path, &rgba, width, height, ColorType::Rgba8)
        .with_context(|| format!("保存首帧失败: {}", path.display()))
}
