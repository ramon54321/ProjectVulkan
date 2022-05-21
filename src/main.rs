use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBuffer, SubpassContents,
    },
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::{BuffersDefinition, VertexInputState},
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    single_pass_renderpass,
    swapchain::{acquire_next_image, Surface, SurfaceCapabilities, Swapchain, SwapchainCreateInfo},
    sync::{now, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn setup_instance() -> Arc<Instance> {
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

fn setup_surface(instance: Arc<Instance>, event_loop: &EventLoop<()>) -> Arc<Surface<Window>> {
    let surface = WindowBuilder::new()
        .with_title("My Vulkan Window")
        .build_vk_surface(event_loop, instance)
        .expect("Unable to create window");
    surface
}

fn setup_logical_device_and_queue(instance: Arc<Instance>) -> (Arc<Device>, Arc<Queue>) {
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

fn setup_render_pass(logical_device: Arc<Device>) -> Arc<RenderPass> {
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
    logical_device: Arc<Device>,
    swap_chain_extent: [u32; 2],
    render_pass: Arc<RenderPass>,
) -> Arc<GraphicsPipeline> {
    mod vertex_shader {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 450

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
                    "
        }
    }

    mod fragment_shader {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
                #version 450

                layout(location = 0) out vec4 outColor;

                void main() {
                    outColor = vec4(0.8, 0.0, 0.0, 1.0);
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
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
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

fn setup_command_buffers(
    logical_device: &Arc<Device>,
    queue: &Arc<Queue>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    graphics_pipeline: &Arc<GraphicsPipeline>,
    vertex_buffer: &Arc<CpuAccessibleBuffer<[Vertex]>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                logical_device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .expect("Could not create command buffer");

            command_buffer_builder
                .begin_render_pass(
                    framebuffer.clone(),
                    SubpassContents::Inline,
                    vec![[1.0, 1.0, 1.0, 1.0].into()],
                )
                .expect("Could not begin render pass")
                .bind_pipeline_graphics(graphics_pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(3, 1, 0, 0)
                .expect("Could not draw")
                .end_render_pass()
                .expect("Could not end render pass");

            let command_buffer = command_buffer_builder
                .build()
                .expect("Could not build command buffer");

            Arc::new(command_buffer)
        })
        .collect()
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod)]
struct Vertex {
    position: [f32; 2],
}
impl_vertex!(Vertex, position);

fn setup_vertexbuffer(logical_device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[Vertex]>> {
    CpuAccessibleBuffer::from_iter(
        logical_device,
        BufferUsage::vertex_buffer(),
        false,
        vec![
            Vertex {
                position: [-0.5, -0.5],
            },
            Vertex {
                position: [0.0, 0.5],
            },
            Vertex {
                position: [0.5, -0.25],
            },
        ]
        .into_iter(),
    )
    .expect("Could not create vertex buffer")
}

fn main_loop(event_loop: EventLoop<()>, vulkan_connector: VulkanConnector) {
    let mut previous_frame_end = Some(now(vulkan_connector.logical_device.clone()));
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::RedrawEventsCleared => {
                previous_frame_end
                    .as_mut()
                    .expect("Could not get previous frame end")
                    .cleanup_finished();

                let command_buffers = setup_command_buffers(
                    &vulkan_connector.logical_device,
                    &vulkan_connector.queue,
                    &vulkan_connector.framebuffers,
                    &vulkan_connector.graphics_pipeline,
                    &vulkan_connector.vertex_buffer,
                );

                let (image_index, is_acquired_image_suboptimal, acquire_future) =
                    match acquire_next_image(vulkan_connector.swapchain.clone(), None) {
                        Ok(result) => result,
                        Err(e) => panic!("{:?}", e),
                    };

                let execution_future = now(vulkan_connector.logical_device.clone())
                    .join(acquire_future)
                    .then_execute(
                        vulkan_connector.queue.clone(),
                        command_buffers[image_index].clone(),
                    )
                    .unwrap()
                    .then_swapchain_present(
                        vulkan_connector.queue.clone(),
                        vulkan_connector.swapchain.clone(),
                        image_index,
                    )
                    .then_signal_fence_and_flush();

                execution_future
                    .expect("Execution future was not present")
                    .wait(None)
                    .expect("Execution future could not wait");
            }
            _ => (),
        }
    });
}

struct VulkanConnector {
    logical_device: Arc<Device>,
    queue: Arc<Queue>,
    framebuffers: Vec<Arc<Framebuffer>>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    swapchain: Arc<Swapchain<Window>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}

fn main() {
    let instance = setup_instance();
    let event_loop = EventLoop::new();
    let surface = setup_surface(instance.clone(), &event_loop);
    let (logical_device, queue) = setup_logical_device_and_queue(instance.clone());
    let (swapchain, images) = setup_swapchain_and_images(logical_device.clone(), surface.clone());
    let render_pass = setup_render_pass(logical_device.clone());
    let graphics_pipeline = setup_graphics_pipeline(
        logical_device.clone(),
        swapchain.image_extent(),
        render_pass.clone(),
    );
    let framebuffers = setup_framebuffers(&images, render_pass.clone());
    let vertex_buffer = setup_vertexbuffer(logical_device.clone());

    let vulkan_connector = VulkanConnector {
        logical_device,
        queue,
        framebuffers,
        graphics_pipeline,
        swapchain,
        vertex_buffer,
    };

    main_loop(event_loop, vulkan_connector);
}
