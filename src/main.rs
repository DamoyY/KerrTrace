extern crate alloc;
mod app;
mod config;
#[cfg(test)]
#[path = "../tests/integration/gpu.rs"]
mod gpu;
mod hud;
#[cfg(test)]
#[path = "../tests/integration/overlay.rs"]
mod overlay;
mod renderer;
#[cfg(test)]
#[path = "../tests/unit/safety.rs"]
mod safety;
use anyhow::Result;
use app::App;
use config::load;
use mimalloc::MiMalloc;
use winit::event_loop::EventLoop;
#[global_allocator]
static GLOBAL_ALLOCATOR: MiMalloc = MiMalloc;
fn main() -> Result<()> {
    env_logger::init();
    let config = load(std::path::Path::new("assets/settings.yaml"))?;
    let event_loop = EventLoop::new()?;
    let mut app = App::new(config);
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.failure {
        return Err(error);
    }
    Ok(())
}
