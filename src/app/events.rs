use super::App;
use alloc::sync::Arc;
use anyhow::Error;
use core::time::Duration;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    window::{WindowAttributes, WindowId},
};
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = WindowAttributes::default()
            .with_title("KerrTrace Rust")
            .with_resizable(false)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.window.width,
                self.config.window.height,
            ));
        match event_loop.create_window(attributes) {
            Ok(window) => {
                self.window = Some(Arc::new(window));
                if let Err(error) = self.init_renderer() {
                    self.fail(event_loop, error);
                }
                self.last_camera_update = Instant::now();
            }
            Err(error) => {
                self.fail(event_loop, error.into());
            }
        }
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if event == WindowEvent::CloseRequested {
            event_loop.exit();
            return;
        }
        if event == WindowEvent::RedrawRequested {
            if self.window.as_ref().is_some_and(|window| {
                let size = window.inner_size();
                size.width == 0 || size.height == 0
            }) {
                return;
            }
            let result = (|| {
                self.update_camera()?;
                self.render()
            })();
            if let Err(error) = result {
                self.fail(event_loop, error);
            }
            return;
        }
        if let WindowEvent::Resized(size) = event {
            if size.width > 0
                && size.height > 0
                && (size.width != self.config.window.width
                    || size.height != self.config.window.height)
            {
                if let Err(error) = self.init_renderer() {
                    self.fail(event_loop, error);
                }
                self.last_rendered = None;
                self.last_camera_update = Instant::now();
            }
            return;
        }
        if let WindowEvent::KeyboardInput {
            device_id: _,
            event: key_event,
            is_synthetic: _,
        } = event
        {
            if let Err(error) = self.handle_keyboard_input(&key_event) {
                self.fail(event_loop, error);
            }
            return;
        }
        if let WindowEvent::MouseInput {
            device_id: _,
            state: ElementState::Pressed,
            button: MouseButton::Left,
        } = event
        {
            if let Err(error) = self.capture_mouse(true) {
                self.fail(event_loop, error);
            }
            return;
        }
        if let WindowEvent::MouseWheel {
            device_id: _,
            delta,
            phase: _,
        } = event
            && self.mouse_locked
        {
            self.handle_mouse_wheel(&delta);
        }
        if event == WindowEvent::Focused(false) {
            self.keys_pressed.clear();
            if self.mouse_locked
                && let Err(error) = self.capture_mouse(false)
            {
                self.fail(event_loop, error);
            }
        }
    }
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.mouse_locked
        {
            self.handle_mouse_motion(delta);
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            self.keys_pressed.clear();
            self.last_camera_update = Instant::now();
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }
        if let Some(limit) = self.config.window.frame_limit {
            let interval = Duration::from_secs_f64(1.0_f64 / f64::from(limit.get()));
            let deadline = self.last_present + interval;
            event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            if Instant::now() < deadline {
                return;
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
        window.request_redraw();
    }
}
impl App {
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: Error) {
        log::error!("{error:#}");
        self.failure = Some(error);
        event_loop.exit();
    }
}
