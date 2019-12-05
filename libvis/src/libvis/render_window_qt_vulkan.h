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


#pragma once

#include <QWindow>

#include "libvis/libvis.h"
#include "libvis/render_window_qt.h"
#include "libvis/vulkan.h"

namespace vis {

// This class is intended to be used as a widget (by using
// QWidget::createWindowContainer(render_window_vulkan)), however itself it
// needs to be a QWindow such that it can get a window handle for use with
// Vulkan.
class RenderWidgetVulkan : public QWindow {
 Q_OBJECT
 public:
  RenderWidgetVulkan();
  
  virtual ~RenderWidgetVulkan();

 public slots:
  void Render();

 protected:
  virtual void resizeEvent(QResizeEvent* event) override;
  
 private slots:
  bool InitializeVulkan();
  
 private:
  void RecreateSwapChain();
  
  void StopAndWaitForRendering();
  bool CreateSurfaceDependentObjects(VkSwapchainKHR old_swap_chain);
  void DestroySurfaceDependentObjects();
  
  // Timer for continuous rendering (will be deleted by Qt).
  QTimer* render_timer_;
  
  // Destructors are called from bottom to top.
  VulkanInstance instance_;
  VulkanPhysicalDevice* selected_physical_device_;
  VulkanDevice device_;
  VkSurfaceKHR surface_;
  VkSwapchainKHR swap_chain_;
  VkFormat swap_chain_image_format_;
  VkExtent2D swap_chain_extent_;
  vector<VkImage> swap_chain_images_;
  vector<VkImageView> swap_chain_image_views_;
  VkRenderPass render_pass_;
  vector<VkFramebuffer> swap_chain_framebuffers_;
  VkCommandPool command_pool_;
  vector<VkCommandBuffer> command_buffers_;
  VkSemaphore image_available_semaphore_;
  VkSemaphore render_finished_semaphore_;
  
  bool vulkan_initialized_;
};

// A Qt and Vulkan based render window implementation for Linux and Windows.
// The actual render widget is a member of this class of type
// RenderWidgetVulkan, and the window is a member of the base class of type
// RenderWindowQtWindow.
class RenderWindowQtVulkan : public RenderWindowQt {
 public:
  RenderWindowQtVulkan(const std::string& title, int width, int height, const shared_ptr<RenderWindowCallbacks>& callbacks);
  
  virtual void RenderFrame() override;

 private:
  // Pointer is managed by Qt.
  RenderWidgetVulkan* render_widget_;
};

}
