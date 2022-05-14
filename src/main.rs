use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct App {}

impl App {
    pub fn new() -> Self {
        Self {}
    }
    pub fn start(&mut self) {
        let event_loop = EventLoop::new();
        let _window = WindowBuilder::new()
            .with_title("My Vulkan Window")
            .build(&event_loop)
            .expect("Unable to create window");
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => (),
            }
        });
    }
}

fn main() {
    let mut app = App::new();
    app.start();
}
