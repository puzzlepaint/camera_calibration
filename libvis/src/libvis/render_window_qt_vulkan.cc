// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include "libvis/render_window_qt_vulkan.h"

#include <unordered_map>

#if defined(VK_USE_PLATFORM_WIN32_KHR)
#include <windows.h>
#elif defined(VK_USE_PLATFORM_XCB_KHR)
#include <QX11Info>
#endif

#include "libvis/logging.h"
#include <QResizeEvent>
#include <QTimer>

#include "libvis/qt_thread.h"

namespace vis {

RenderWidgetVulkan::RenderWidgetVulkan()
    : QWindow(),
      vulkan_initialized_(false) {}

RenderWidgetVulkan::~RenderWidgetVulkan() {
  StopAndWaitForRendering();
  DestroySurfaceDependentObjects();
  
  vkDestroyCommandPool(device_.device(), command_pool_, nullptr);
  vkDestroySwapchainKHR(device_.device(), swap_chain_, nullptr);
  vkDestroySemaphore(device_.device(), image_available_semaphore_, nullptr);
  vkDestroySemaphore(device_.device(), render_finished_semaphore_, nullptr);
  vkDestroySurfaceKHR(instance_.instance(), surface_, nullptr);
}

void RenderWidgetVulkan::Render() {
  // Asynchronously get the next image from the swap chain.
  // image_available_semaphore_ will be signaled when the image becomes
  // available. The index refers to the swap_chain_images_ array.
  u32 image_index;
  VkResult result = vkAcquireNextImageKHR(
      device_.device(), swap_chain_,
      /* timeout */ numeric_limits<uint64_t>::max(),
      image_available_semaphore_, VK_NULL_HANDLE, &image_index);
  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    RecreateSwapChain();
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    LOG(ERROR) << "Failed to acquire swap chain image.";
    return;
  }
  
  // Submit the right command buffer for the image.
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  // Wait with the color attachment output until the image_available_semaphore_
  // was signaled.
  submit_info.waitSemaphoreCount = 1;
  VkSemaphore wait_semaphores[] = {image_available_semaphore_};
  submit_info.pWaitSemaphores = wait_semaphores;
  // Could use VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, but then one must
  // create a dependency for the render pass to have the swap chain image
  // acquired, otherwise the image format transition at the start of the render
  // pass cannot work.
  VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
  submit_info.pWaitDstStageMask = wait_stages;
  // Specify the command buffer(s) to submit.
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffers_[image_index];
  // Signal the render_finished_semaphore_ after the command buffers finished.
  VkSemaphore signal_semaphores[] = {render_finished_semaphore_};
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;
  if (vkQueueSubmit(device_.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS) {
    LOG(ERROR) << "Failed to submit draw command buffer.";
  }
  
  // Put the image back into the swap chain for presentation.
  VkPresentInfoKHR present_info = {};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;
  
  VkSwapchainKHR swap_chains[] = {swap_chain_};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swap_chains;
  present_info.pImageIndices = &image_index;
  // Only useful if using more than one swap chain.
  present_info.pResults = nullptr;
  
  result = vkQueuePresentKHR(device_.presentation_queue(), &present_info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    RecreateSwapChain();
  } else if (result != VK_SUCCESS) {
    LOG(ERROR) << "Failed to present swap chain image.";
    return;
  }
}

void RenderWidgetVulkan::resizeEvent(QResizeEvent* event) {
  if (event->size().width() == 0 || event->size().height() == 0) {
    return;
  }
  
  if (!vulkan_initialized_) {
    InitializeVulkan();
  } else {
    RecreateSwapChain();
  }
}

bool RenderWidgetVulkan::InitializeVulkan() {
  // Enable debug layers in debug builds only. CMake's RelWithDebInfo defines
  // NDEBUG by default and will therefore not enable debug layers.
#ifdef NDEBUG
  constexpr bool kEnableDebugLayers = false;
#else
  constexpr bool kEnableDebugLayers = true;
#endif
  
  vector<string> instance_extensions;
  vector<string> device_extensions;
  
  // Choose the instance extensions to enable.
  // Since this Vulkan instance is used for rendering with the output being
  // presented on the screen, we need to enable the surface extensions.
  instance_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#if VK_USE_PLATFORM_WIN32_KHR
  instance_extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif VK_USE_PLATFORM_XCB_KHR
  instance_extensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#else
  LOG(ERROR) << "No supported surface extension for this platform.";
  return false;
#endif
  
  // For the same reason, the swap chain device extension is needed.
  device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  
  // Initialize the Vulkan instance.
  if (!instance_.Initialize(instance_extensions, kEnableDebugLayers)) {
    return false;
  }
  
  // Create the VkSurfaceKHR, which represents the surface to present rendered
  // images to. This should be done before the physical device selection because
  // some devices may not be able to present to the created surface.
#if VK_USE_PLATFORM_WIN32_KHR
  VkWin32SurfaceCreateInfoKHR create_info = {};
  create_info.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  create_info.hinstance = GetModuleHandle(nullptr);
  create_info.hwnd      = reinterpret_cast<HWND> (this->winId());
  auto vkCreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR) vkGetInstanceProcAddr(instance_.instance(), "vkCreateWin32SurfaceKHR");
  if (!vkCreateWin32SurfaceKHR) {
    LOG(ERROR) << "vkCreateWin32SurfaceKHR() not available.";
    return false;
  }
  if (vkCreateWin32SurfaceKHR(instance_.instance(), &create_info, nullptr, &surface_) != VK_SUCCESS) {
    LOG(ERROR) << "vkCreateWin32SurfaceKHR() failed.";
    return false;
  }
#elif VK_USE_PLATFORM_XCB_KHR
  VkXcbSurfaceCreateInfoKHR create_info = {};
  create_info.sType      = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
  create_info.connection = QX11Info::connection();
  create_info.window     = static_cast<xcb_window_t>(this->winId());
  auto vkCreateXcbSurfaceKHR = (PFN_vkCreateXcbSurfaceKHR) vkGetInstanceProcAddr(instance_.instance(), "vkCreateXcbSurfaceKHR");
  if (!vkCreateXcbSurfaceKHR) {
    LOG(ERROR) << "vkCreateXcbSurfaceKHR() not available.";
    return false;
  }
  if (vkCreateXcbSurfaceKHR(instance_.instance(), &create_info, nullptr, &surface_) != VK_SUCCESS) {
    LOG(ERROR) << "vkCreateXcbSurfaceKHR() failed.";
    return false;
  }
#endif
  
  // Find the best suited physical device.
  // TODO: The scoring of the devices should be also be influenced by some model
  //       of the renderer that will be used, which may request certain
  //       features.
  std::multimap<float, VulkanPhysicalDevice*> scored_devices;
  // Holds the index of a suitable queue family for each device.
  std::unordered_map<VulkanPhysicalDevice*, u32> selected_graphics_queue_family_indices;
  std::unordered_map<VulkanPhysicalDevice*, u32> selected_presentation_queue_family_indices;
  for (const shared_ptr<VulkanPhysicalDevice>& device : instance_.physical_devices()) {
    // Check for support of the required queues.
    bool has_graphics_queue = false;
    bool has_presentation_queue = false;
    for (u32 queue_family_index = 0; queue_family_index < device->queue_families().size(); ++ queue_family_index) {
      const VkQueueFamilyProperties& family_properties = device->queue_families().at(queue_family_index);
      
      // NOTE: For compatibility reasons, it may be good not to require a queue
      //       count higher than 1: On an Intel Ivy Bridge integrated GPU, only
      //       one queue is available (using the driver available to me at the
      //       time of writing, January 2017).
      if (family_properties.queueCount > 0 && family_properties.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        selected_graphics_queue_family_indices[device.get()] = queue_family_index;
        has_graphics_queue = true;
      }
      VkBool32 presentation_support = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device->device(), queue_family_index, surface_, &presentation_support);
      if (family_properties.queueCount > 0 && presentation_support) {
        selected_presentation_queue_family_indices[device.get()] = queue_family_index;
        has_presentation_queue = true;
      }
      
      // If we found a queue which supports both graphics and presentation,
      // do an early exit.
      if (has_graphics_queue && has_presentation_queue &&
          selected_graphics_queue_family_indices[device.get()] ==
              selected_presentation_queue_family_indices[device.get()]) {
        break;
      }
    }
    if (!has_graphics_queue || !has_presentation_queue) {
      continue;
    }
    
    // Check for support of the required extensions.
    float device_score = 0;
    for (const string& requested_extension : device_extensions) {
      bool found = false;
      for (const VkExtensionProperties& extension_properties : device->available_extensions()) {
        if (requested_extension == extension_properties.extensionName) {
          found = true;
          break;
        }
      }
      if (!found) {
        device_score = -1;
        break;
      }
    }
    if (device_score < 0) {
      continue;
    }
    
    // Check for support of the required swap chain details given our surface.
    // This query must be done after the extensions check (because the swap
    // chain extension might not be available).
    VulkanSwapChainSupport swap_chain_support;
    device->QuerySwapChainSupport(surface_, &swap_chain_support);
    bool swap_chain_support_ok = !swap_chain_support.formats.empty() &&
                                 !swap_chain_support.present_modes.empty();
    if (!swap_chain_support_ok) {
      continue;
    }
    
    // Score the device according to its type.
    if (device->properties().deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      device_score += 10;
    } else if (device->properties().deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      device_score += 8;
    } else if (device->properties().deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
      device_score += 6;
    } else if (device->properties().deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
      device_score += 4;
    }
    
    scored_devices.insert(make_pair(device_score, device.get()));
  }
  if (scored_devices.empty()) {
    LOG(ERROR) << "No suitable Vulkan device found.";
    return false;
  }
  selected_physical_device_ = scored_devices.crbegin()->second;
  u32 selected_graphics_queue_family_index = selected_graphics_queue_family_indices.at(selected_physical_device_);
  u32 selected_presentation_queue_family_index = selected_presentation_queue_family_indices.at(selected_physical_device_);
  
  // Create a logical device for the selected physical device.
  if (!device_.Initialize(selected_physical_device_->device(),
                          device_extensions,
                          selected_graphics_queue_family_index,
                          selected_presentation_queue_family_index,
                          instance_)) {
    return false;
  }
  
  // Create command pool.
  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = selected_graphics_queue_family_index;
  // Possible flags:
  // - VK_COMMAND_POOL_CREATE_TRANSIENT_BIT:
  //     Hint that command buffers are re-recorded very often.
  // - VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT:
  //     Allows command buffers to be re-recorded individually.
  pool_info.flags = 0;
  if (vkCreateCommandPool(device_.device(), &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
    LOG(ERROR) << "Failed to create a command pool.";
    return false;
  }
  
  // Create semaphores.
  VkSemaphoreCreateInfo semaphore_info = {};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  if (vkCreateSemaphore(device_.device(), &semaphore_info, nullptr, &image_available_semaphore_) != VK_SUCCESS ||
      vkCreateSemaphore(device_.device(), &semaphore_info, nullptr, &render_finished_semaphore_) != VK_SUCCESS) {
    LOG(ERROR) << "Failed to create a semaphore.";
    return false;
  }
  
  // Create surface dependent objects.
  if (!CreateSurfaceDependentObjects(VK_NULL_HANDLE)) {
    return false;
  }
  
  vulkan_initialized_ = true;
  
  // Start rendering timer for continuous rendering.
  render_timer_ = new QTimer(this);
  render_timer_->setInterval(1);
  connect(render_timer_, &QTimer::timeout, this, &RenderWidgetVulkan::Render);
  render_timer_->start();
  
  return true;
}

void RenderWidgetVulkan::RecreateSwapChain() {
  StopAndWaitForRendering();
  // TODO: It seems like a more elaborate solution to find the objects to
  //       recreate could pay off here. For example, some dependent objects
  //       may only need to be recreated if the swap chain's format changes,
  //       which is extremely unlikely.
  DestroySurfaceDependentObjects();
  CreateSurfaceDependentObjects(swap_chain_);
  
  // Restart rendering.
  // TODO: Should not be done for event-based rendering.
  render_timer_->start();
}

void RenderWidgetVulkan::StopAndWaitForRendering() {
  render_timer_->stop();
  vkDeviceWaitIdle(device_.device());
}

bool RenderWidgetVulkan::CreateSurfaceDependentObjects(VkSwapchainKHR old_swap_chain) {
  // Gather information for creating a swap chain.
  VulkanSwapChainSupport swap_chain_support;
  selected_physical_device_->QuerySwapChainSupport(surface_, &swap_chain_support);
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Selected swap chain:";
    VLOG(1) << "  minImageCount: " << swap_chain_support.capabilities.minImageCount;
    VLOG(1) << "  maxImageCount: " << swap_chain_support.capabilities.maxImageCount;
    VLOG(1) << "  currentExtent: (" << swap_chain_support.capabilities.currentExtent.width << ", " << swap_chain_support.capabilities.currentExtent.height << ")";
    VLOG(1) << "  minImageExtent: (" << swap_chain_support.capabilities.minImageExtent.width << ", " << swap_chain_support.capabilities.minImageExtent.height << ")";
    VLOG(1) << "  maxImageExtent: (" << swap_chain_support.capabilities.maxImageExtent.width << ", " << swap_chain_support.capabilities.maxImageExtent.height << ")";
    VLOG(1) << "  maxImageArrayLayers: " << swap_chain_support.capabilities.maxImageArrayLayers;
    VLOG(1) << "  supportedTransforms: " << swap_chain_support.capabilities.supportedTransforms;
    VLOG(1) << "  currentTransform: " << swap_chain_support.capabilities.currentTransform;
    VLOG(1) << "  supportedCompositeAlpha: " << swap_chain_support.capabilities.supportedCompositeAlpha;
    VLOG(1) << "  supportedUsageFlags: " << swap_chain_support.capabilities.supportedUsageFlags;
    VLOG(1) << "  #supported formats: " << swap_chain_support.formats.size();
    string present_mode_names;
    for (VkPresentModeKHR mode : swap_chain_support.present_modes) {
      if (!present_mode_names.empty()) {
        present_mode_names += ", ";
      }
      if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        present_mode_names += "immediate";
      } else if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        present_mode_names += "mailbox";
      } else if (mode == VK_PRESENT_MODE_FIFO_KHR) {
        present_mode_names += "fifo";
      } else if (mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
        present_mode_names += "fifo_relaxed";
      }
    }
    VLOG(1) << "  supported present modes: " << present_mode_names;
  }
  
  // Find the best available surface format. This determines the way colors are
  // represented. We prefer a standard 32 bits-per-pixel unsigned normalized
  // format in BGRA pixel ordering (the corresponding RGBA format wasn't
  // available on an Intel IvyBridge mobile GPU), and sRGB color space.
  // NOTE: The only other available format on the IvyBridge mobile GPU is:
  // VK_FORMAT_R8G8B8A8_SRGB.
  // TODO: Test this and see how it differs.
  VkSurfaceFormatKHR preferred_surface_format;
  preferred_surface_format.format = VK_FORMAT_B8G8R8A8_UNORM;
  preferred_surface_format.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  VkSurfaceFormatKHR selected_surface_format;
  if (swap_chain_support.formats.size() == 1 &&
      swap_chain_support.formats[0].format == VK_FORMAT_UNDEFINED) {
    // Any format is allowed. Choose our preferred one.
    selected_surface_format = preferred_surface_format;
  } else {
    // Go through the list of supported formats to see if the preferred format
    // is supported.
    bool found = false;
    for (const auto& available_format : swap_chain_support.formats) {
      if (available_format.format == preferred_surface_format.format &&
          available_format.colorSpace == preferred_surface_format.colorSpace) {
        selected_surface_format = preferred_surface_format;
        found = true;
        break;
      }
    }
    if (!found) {
      // The preferred surface format is not available. Simply choose the first
      // available one.
      // TODO: Could rank the available formats and choose the best.
      LOG(WARNING) << "The preferred surface format for the swap chain is not available. Choosing format: " << swap_chain_support.formats[0].format;
      LOG(WARNING) << "Available formats:";
      for (const auto& available_format : swap_chain_support.formats) {
        LOG(WARNING) << "  VkFormat " << available_format.format << " VkColorSpaceKHR " << available_format.colorSpace;
      }
      selected_surface_format = swap_chain_support.formats[0];
    }
  }
  swap_chain_image_format_ = selected_surface_format.format;
  
  // Find the best available presentation mode.
  // Out of the 4 modes offered by Vulkan, 2 are interesing for us (since they
  // are the ones to avoid tearing):
  // VK_PRESENT_MODE_FIFO_KHR: The swap chain is a queue. On a vertical blank,
  //     the first image is taken and displayed. If the queue is full, the
  //     application has to wait. Similar to VSync. This mode is guaranteed to
  //     be supported.
  // VK_PRESENT_MODE_MAILBOX_KHR: Differing from the mode above, if the queue is
  //     full, new images can replace the existing ones. Can be used to
  //     implement triple buffering.
  // Policy: Choose VK_PRESENT_MODE_MAILBOX_KHR is available, otherwise
  // VK_PRESENT_MODE_FIFO_KHR.
  VkPresentModeKHR selected_present_mode = VK_PRESENT_MODE_FIFO_KHR;
  for (VkPresentModeKHR present_mode : swap_chain_support.present_modes) {
    if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      selected_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
      break;
    }
  }
  
  // Choose the resolution of the swap chain images equal to the window size,
  // if possible.
  if (swap_chain_support.capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    // The extent that shall be used is given.
    swap_chain_extent_ = swap_chain_support.capabilities.currentExtent;
  } else {
    // We can choose the extent ourselves.
    swap_chain_extent_ = {static_cast<u32>(width()), static_cast<u32>(height())};
    swap_chain_extent_.width =
        std::max(swap_chain_support.capabilities.minImageExtent.width,
                  std::min(swap_chain_support.capabilities.maxImageExtent.width,
                          swap_chain_extent_.width));
    swap_chain_extent_.height =
        std::max(swap_chain_support.capabilities.minImageExtent.height,
                  std::min(swap_chain_support.capabilities.maxImageExtent.height,
                          swap_chain_extent_.height));
  }
  
  // Decide on the number of images in the swap chain.
  // Policy: If VK_PRESENT_MODE_MAILBOX_KHR is available, use triple buffering:
  // Ideally, one frame is used for display, while one is being rendered to, and
  // the third frame is ready for display. Thus, the frame which is ready can be
  // updated until it is put on display. Con: frames may be rendered and never
  // shown; It is not clear which frame will be displayed in the end, so it is
  // not clear at which time the application state should be rendered if
  // something is moving, so movements will not be completely fluid. If
  // VK_PRESENT_MODE_FIFO_KHR is used, use double buffering to
  // reduce the latency.
  u32 image_count =
      (selected_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) ? 3 : 2;
  image_count = std::max(image_count, swap_chain_support.capabilities.minImageCount);
  if (swap_chain_support.capabilities.maxImageCount > 0 &&
      image_count > swap_chain_support.capabilities.maxImageCount) {
    image_count = swap_chain_support.capabilities.maxImageCount;
  }
  
  // Create the swap chain.
  VkSwapchainCreateInfoKHR swap_chain_create_info = {};
  swap_chain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swap_chain_create_info.surface = surface_;
  swap_chain_create_info.minImageCount = image_count;
  swap_chain_create_info.imageFormat = selected_surface_format.format;
  swap_chain_create_info.imageColorSpace = selected_surface_format.colorSpace;
  swap_chain_create_info.imageExtent = swap_chain_extent_;
  swap_chain_create_info.imageArrayLayers = 1;
  // Potentially use VK_IMAGE_USAGE_TRANSFER_DST_BIT here if the scene is
  // rendered to a different image first and then transferred to the output:
  swap_chain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  
  u32 shared_queue_family_indices[] = {static_cast<u32>(device_.graphics_queue_family_index()),
                                       static_cast<u32>(device_.presentation_queue_family_index())};
  if (device_.graphics_queue_family_index() != device_.presentation_queue_family_index()) {
    // NOTE: VK_SHARING_MODE_EXCLUSIVE would also be possible using explicit
    //       ownership transfers.
    swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    swap_chain_create_info.queueFamilyIndexCount = 2;
    swap_chain_create_info.pQueueFamilyIndices = shared_queue_family_indices;
  } else {
    swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swap_chain_create_info.queueFamilyIndexCount = 0;
    swap_chain_create_info.pQueueFamilyIndices = nullptr;
  }
  
  // Do not use any special transforms. At the time of writing (January 2017),
  // the potentially available transformations are rotation and mirroring of
  // the image.
  swap_chain_create_info.preTransform = swap_chain_support.capabilities.currentTransform;
  // Do not use the alpha channel to blend with other windows in the window
  // system.
  swap_chain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swap_chain_create_info.presentMode = selected_present_mode;
  // Do not care about the colors of pixels that are obscured (for example by
  // other windows in the window system).
  swap_chain_create_info.clipped = VK_TRUE;
  swap_chain_create_info.oldSwapchain = old_swap_chain;
  
  if (vkCreateSwapchainKHR(device_.device(), &swap_chain_create_info, nullptr, &swap_chain_) != VK_SUCCESS) {
    LOG(ERROR) << "Swap chain creation failed.";
    return false;
  }
  
  if (old_swap_chain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device_.device(), old_swap_chain, nullptr);
  }
  
  // Get the handles of the images in the swap chain.
  // NOTE: image_count passed into swap_chain_create_info.minImageCount only
  //       specifies the minimum image count. More images could have been
  //       actually created.
  vkGetSwapchainImagesKHR(device_.device(), swap_chain_, &image_count, nullptr);
  VLOG(1) << "Swap chain image count: " << image_count;
  swap_chain_images_.resize(image_count);
  vkGetSwapchainImagesKHR(device_.device(), swap_chain_, &image_count, swap_chain_images_.data());
  
  // Create image views for each image in the swap chain.
  swap_chain_image_views_.resize(swap_chain_images_.size());
  for (usize i = 0; i < swap_chain_image_views_.size(); ++ i) {
    VkImageViewCreateInfo image_view_create_info = {};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.image = swap_chain_images_[i];
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = swap_chain_image_format_;
    // Do not use any color channel swizzling or constant 0 / 1 mapping.
    image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = 0;
    image_view_create_info.subresourceRange.levelCount = 1;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_.device(), &image_view_create_info, nullptr, &swap_chain_image_views_[i]) != VK_SUCCESS) {
      LOG(ERROR) << "Failed to create an image view.";
      return false;
    }
  }
  
  // Create render subpass.
  VkAttachmentReference color_attachment_ref = {};
  // This references the index in render_pass_info.pAttachments.
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  // The index of the attachment in this array will be referenced from shaders,
  // for example: layout(location = 0) out vec4 out_color
  subpass.pColorAttachments = &color_attachment_ref;
  
  // Create render pass.
  VkAttachmentDescription color_attachment = {};
  color_attachment.format = swap_chain_image_format_;
  // Do not use multisampling:
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  // Alternatives:
  // - VK_ATTACHMENT_LOAD_OP_LOAD:      Retain the previous content.
  // - VK_ATTACHMENT_LOAD_OP_DONT_CARE: Start with undefined content.
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  // Ignore the stencil buffer:
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  // Do not care about the previous image format. This prevents preserving the
  // contents, but this does not matter here as the image is cleared.
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  // The image should be ready for presentation after rendering.
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  
  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  if (vkCreateRenderPass(device_.device(), &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS) {
    LOG(ERROR) << "Failed to create a render pass.";
    return false;
  }
  
  // Create a framebuffer for each swap chain image to be able to bind it as an
  // attachment to the render pass. A framebuffer references all attachments,
  // but here we only use one color image.
  swap_chain_framebuffers_.resize(swap_chain_image_views_.size());
  for (usize i = 0; i < swap_chain_image_views_.size(); ++ i) {
    VkImageView attachments[] = {swap_chain_image_views_[i]};

    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass_;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = attachments;
    framebuffer_info.width = swap_chain_extent_.width;
    framebuffer_info.height = swap_chain_extent_.height;
    framebuffer_info.layers = 1;

    if (vkCreateFramebuffer(device_.device(), &framebuffer_info, nullptr, &swap_chain_framebuffers_[i]) != VK_SUCCESS) {
      LOG(ERROR) << "Failed to create a framebuffer.";
      return false;
    }
  }
  
  // Create command buffers: one for each framebuffer.
  command_buffers_.resize(swap_chain_framebuffers_.size());
  VkCommandBufferAllocateInfo command_buffers_info = {};
  command_buffers_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffers_info.commandPool = command_pool_;
  // Primary buffers can be submitted directly, while secondary buffers can be
  // called from primary buffers.
  command_buffers_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffers_info.commandBufferCount = command_buffers_.size();
  if (vkAllocateCommandBuffers(device_.device(), &command_buffers_info, command_buffers_.data()) != VK_SUCCESS) {
    LOG(ERROR) << "Failed to allocate command buffers.";
    return false;
  }
  
  for (size_t i = 0; i < command_buffers_.size(); i++) {
    // Begin recording.
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // Possible values:
    // - VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT indicates that each
    //   recording of the command buffer will only be submitted once, and the
    //   command buffer will be reset and recorded again between each
    //   submission.
    // - VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT indicates that a
    //   secondary command buffer is considered to be entirely inside a render
    //   pass. If this is a primary command buffer, then this bit is ignored.
    // - Setting VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT allows the command
    //   buffer to be resubmitted to a queue or recorded into a primary command
    //   buffer while it is pending execution.
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    begin_info.pInheritanceInfo = nullptr;
    // This resets a buffer if it had been recorded before.
    vkBeginCommandBuffer(command_buffers_[i], &begin_info);
    
    // Start the render pass.
    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = swap_chain_framebuffers_[i];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swap_chain_extent_;
    VkClearValue clear_color;
    clear_color.color.float32[0] = 0.8f;
    clear_color.color.float32[1] = 0.2f;
    clear_color.color.float32[2] = 0.2f;
    clear_color.color.float32[3] = 1.0f;
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_color;
    // If contents is VK_SUBPASS_CONTENTS_INLINE, the contents of the subpass
    // will be recorded inline in the primary command buffer, and secondary
    // command buffers must not be executed within the subpass. If contents is
    // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS, the contents are recorded
    // in secondary command buffers that will be called from the primary command
    // buffer, and vkCmdExecuteCommands is the only valid command on the command
    // buffer until vkCmdNextSubpass or vkCmdEndRenderPass.
    vkCmdBeginRenderPass(command_buffers_[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    
    // No actual rendering commands yet. They would be placed here.
    
    // End the render pass.
    vkCmdEndRenderPass(command_buffers_[i]);
    
    // End recording.
    if (vkEndCommandBuffer(command_buffers_[i]) != VK_SUCCESS) {
      LOG(ERROR) << "Failed to record a command buffer.";
      return false;
    }
  }
  
  // TODO: Continue with shaders. #include "shader.frag.h" and #include "shader.vert.h" from the build directory.
  
  return true;
}

void RenderWidgetVulkan::DestroySurfaceDependentObjects() {
  vkFreeCommandBuffers(device_.device(), command_pool_, command_buffers_.size(), command_buffers_.data());
  for (const VkFramebuffer& framebuffer : swap_chain_framebuffers_) {
    vkDestroyFramebuffer(device_.device(), framebuffer, nullptr);
  }
  vkDestroyRenderPass(device_.device(), render_pass_, nullptr);
  for (const VkImageView& image_view : swap_chain_image_views_) {
    vkDestroyImageView(device_.device(), image_view, nullptr);
  }
}

RenderWindowQtVulkan::RenderWindowQtVulkan(const string& title, int width, int height, const shared_ptr<RenderWindowCallbacks>& callbacks)
    : RenderWindowQt(title, width, height, callbacks) {
  QtThread::Instance()->RunInQtThreadBlocking([&](){
    // Add the Vulkan render widget to the window created by the parent class.
    render_widget_ = new RenderWidgetVulkan();
    window_->setCentralWidget(QWidget::createWindowContainer(render_widget_));
  });
}

void RenderWindowQtVulkan::RenderFrame() {
  // TODO
}

}
