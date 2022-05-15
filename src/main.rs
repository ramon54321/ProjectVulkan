use std::sync::Arc;
use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo,
    },
    image::{ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    swapchain::{Surface, SurfaceCapabilities, Swapchain, SwapchainCreateInfo},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

struct App {}

impl App {
    pub fn new() -> Self {
        Self {}
    }

    pub fn start(&mut self) {
        let instance = self.setup_instance();
        let event_loop = EventLoop::new();
        let surface = self.setup_surface(instance.clone(), &event_loop);
        let (logical_device, queue) = self.setup_logical_device_and_queue(instance.clone());
        let (swapchain, images) =
            self.setup_swapchain_and_images(logical_device.clone(), surface.clone());
        self.main_loop(event_loop);
    }

    fn setup_instance(&mut self) -> Arc<Instance> {
        let instance = Instance::new(InstanceCreateInfo {
            enabled_extensions: InstanceExtensions {
                khr_get_physical_device_properties2: true,
                khr_surface: true,
                mvk_macos_surface: true,
                ..InstanceExtensions::none()
            },
            ..Default::default()
        })
        .expect("Could not create instance");
        instance
    }

    fn setup_surface(
        &mut self,
        instance: Arc<Instance>,
        event_loop: &EventLoop<()>,
    ) -> Arc<Surface<Window>> {
        let surface = WindowBuilder::new()
            .with_title("My Vulkan Window")
            .build_vk_surface(event_loop, instance)
            .expect("Unable to create window");
        surface
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
                    khr_swapchain: true,
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

    fn setup_swapchain_and_images(
        &mut self,
        logical_device: Arc<Device>,
        surface: Arc<Surface<Window>>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        // Just for demonstration purposes on querying capabilities
        //let capabilities = logical_device
        //.physical_device()
        //.surface_capabilities(&surface, Default::default());
        //println!("{:?}", capabilities);
        let image_usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };
        let swapchain_create_info = SwapchainCreateInfo {
            image_usage: image_usage,
            ..SwapchainCreateInfo::default()
        };
        let (swapchain, images) = Swapchain::new(logical_device, surface, swapchain_create_info)
            .expect("Could not create swapchain");
        (swapchain, images)
    }

    fn main_loop(&mut self, event_loop: EventLoop<()>) {
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
