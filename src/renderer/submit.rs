use super::CudaRenderer;
use anyhow::{Context as _, Result};
use cudarc::driver::PushKernelArg as _;
impl CudaRenderer {
    pub fn submit_render(
        &mut self,
        position: [f32; 3],
        forward: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        fov_scale: f32,
    ) -> Result<bool> {
        let Some(frame) = self.available.front_mut() else {
            return Ok(false);
        };
        self.stream.memset_zeros(&mut frame.device_error)?;
        let mut trace = self.stream.launch_builder(&self.trace_kernel);
        trace
            .arg(&mut frame.hdr_buffer)
            .arg(&self.width)
            .arg(&self.height);
        let vectors = [position, forward, right, up];
        for vector in &vectors {
            for component in vector {
                trace.arg(component);
            }
        }
        trace
            .arg(&self.lut_texture.texture)
            .arg(&self.lut_size)
            .arg(&self.lut_max_temp)
            .arg(&mut frame.device_error)
            .arg(&self.disk_texture.texture)
            .arg(&self.disk_inner)
            .arg(&self.disk_outer)
            .arg(&fov_scale);
        unsafe { trace.launch(self.launch_config) }.context("启动 trace kernel 失败")?;
        self.stream
            .memcpy_dtoh(&frame.device_error, &mut frame.host_error)?;
        if self.bloom_active != 0_i32 {
            let mut bloom = self.stream.launch_builder(&self.bloom_kernel);
            bloom
                .arg(&frame.hdr_buffer)
                .arg(&mut frame.bloom_buffer)
                .arg(&self.width)
                .arg(&self.height)
                .arg(&self.bloom_radius)
                .arg(&self.bloom_sigma)
                .arg(&self.bloom_active);
            unsafe { bloom.launch(self.launch_config) }.context("启动 bloom kernel 失败")?;
        }
        let mut post = self.stream.launch_builder(&self.post_kernel);
        post.arg(&frame.hdr_buffer)
            .arg(&frame.bloom_buffer)
            .arg(&mut frame.image_gpu)
            .arg(&self.width)
            .arg(&self.height)
            .arg(&self.bloom_radius)
            .arg(&self.bloom_sigma)
            .arg(&self.bloom_intensity)
            .arg(&self.bloom_active);
        unsafe { post.launch(self.post_launch_config) }.context("启动 post process kernel 失败")?;
        self.stream
            .memcpy_dtoh(&frame.image_gpu, &mut frame.host_image)?;
        frame.ready_event.record(&self.stream)?;
        let submitted = self.available.pop_front().context("可用帧队列状态异常")?;
        self.pending.push_back(submitted);
        Ok(true)
    }
}
