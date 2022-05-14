use std::sync::Arc;
use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
};
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
        let instance = self.setup_instance();
        let (logical_device, queue) = self.setup_logical_device_and_queue(instance);
        self.main_loop();
    }
    fn setup_instance(&mut self) -> Arc<Instance> {
        let instance = Instance::new(InstanceCreateInfo {
            enabled_extensions: InstanceExtensions {
                khr_get_physical_device_properties2: true,
                ..InstanceExtensions::none()
            },
            ..Default::default()
        })
        .expect("Could not create instance");
        instance
    }
    fn setup_logical_device_and_queue(
        &mut self,
        instance: Arc<Instance>,
    ) -> (Arc<Device>, Arc<Queue>) {
        let physical_device = PhysicalDevice::enumerate(&instance)
            .next()
            .expect("Could not find physical device");
        let queue_family = physical_device
            .queue_families()
            .find(|queue_family| queue_family.supports_graphics())
            .expect("Could not find queue family which supports graphics");
        let (logical_device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    khr_portability_subset: true,
                    ..DeviceExtensions::none()
                },
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                ..Default::default()
            },
        )
        .expect("Could not create logical device");
        let queue = queues.next().expect("Could not get first queue");
        (logical_device, queue)
    }
    fn main_loop(&mut self) {
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
