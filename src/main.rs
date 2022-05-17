use std::sync::Arc;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBuffer,
    },
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    single_pass_renderpass,
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
        let render_pass = self.setup_render_pass(logical_device.clone());
        let graphics_pipeline = self.setup_graphics_pipeline(
            logical_device.clone(),
            swapchain.image_extent(),
            render_pass.clone(),
        );
        let framebuffers = self.setup_framebuffers(&images, render_pass.clone());
        let command_buffer = self.setup_command_buffer(logical_device.clone(), queue.clone());
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

    fn setup_render_pass(&mut self, logical_device: Arc<Device>) -> Arc<RenderPass> {
        single_pass_renderpass!(logical_device,
                                attachments: {
                                    color: {
                                        load: Clear,
                                        store: Store,
                                        format: Format::B8G8R8A8_UNORM,
                                        samples: 1,
                                    }
                                },
                                pass: {
                                    color: [color],
                                    depth_stencil: {}
                                }
        )
        .unwrap()
    }

    fn setup_graphics_pipeline(
        &mut self,
        logical_device: Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        mod vertex_shader {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: "
                #version 450

                layout(location = 0) out vec3 fragColor;

                vec2 positions[3] = vec2[](
                    vec2(0.0, -0.5),
                    vec2(0.5, 0.5),
                    vec2(-0.5, 0.5)
                );

                vec3 colors[3] = vec3[](
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 0.0, 1.0)
                );

                void main() {
                    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
                    fragColor = colors[gl_VertexIndex];
                }
                    "
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: "
                #version 450

                layout(location = 0) in vec3 fragColor;

                layout(location = 0) out vec4 outColor;

                void main() {
                    outColor = vec4(fragColor, 1.0);
                }
                    "
            }
        }

        let vertex_shader_module =
            vertex_shader::load(logical_device.clone()).expect("Could not load vertex shader");
        let fragment_shader_module =
            fragment_shader::load(logical_device.clone()).expect("Could not load fragment shader");

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [swap_chain_extent[0] as f32, swap_chain_extent[1] as f32],
            depth_range: 0.0..1.0,
        };

        let pipeline_builder = GraphicsPipeline::start()
            .vertex_input_state(VertexInputState::default())
            .input_assembly_state(InputAssemblyState::default())
            .vertex_shader(
                vertex_shader_module
                    .entry_point("main")
                    .expect("Could not find entry point for vertex shader module"),
                (),
            )
            .fragment_shader(
                fragment_shader_module
                    .entry_point("main")
                    .expect("Could not find entry point for fragment shader module"),
                (),
            )
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            .render_pass(
                Subpass::from(render_pass.clone(), 0)
                    .expect("Could not create subpass from render pass"),
            );

        pipeline_builder
            .build(logical_device.clone())
            .expect("Could not build graphics pipeline")
    }

    fn setup_framebuffers(
        &mut self,
        images: &Vec<Arc<SwapchainImage<Window>>>,
        render_pass: Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let image_view =
                    ImageView::new_default(image.clone()).expect("Could not create image view");
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![image_view],
                        ..FramebufferCreateInfo::default()
                    },
                )
                .expect("Could not create framebuffer")
            })
            .collect()
    }

    fn setup_command_buffer(
        &mut self,
        logical_device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            logical_device,
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Could not create command buffer")
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
