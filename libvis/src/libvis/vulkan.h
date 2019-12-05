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

#include <memory>
#include <string>
#include <vector>

// Before including Vulkan's header, specify which platform to use.
#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR    1
#elif defined(__linux)
#define VK_USE_PLATFORM_XCB_KHR      1
#else
#error Platform not supported
#endif
#include <vulkan/vulkan.h>

#include "libvis/libvis.h"

namespace vis {

class VulkanPhysicalDevice;

// Wraps a VkInstance and related functionality.
// 
// After constructing an object of this class, Initialize() must be called
// first. If this returns success, physical_devices() can be called to get a
// list of available devices which support Vulkan. Then, the application must
// select one (or more) suitable device(s) based on which features it requires.
class VulkanInstance {
 public:
  // This does not initialize Vulkan yet. Initialize() must be called for that.
  VulkanInstance();
  
  // Destroys the instance.
  ~VulkanInstance();
  
  // Initializes the instance, returns true on success. If not all of the
  // requested extensions are available, initialization fails.
  bool Initialize(const vector<string>& extensions, bool enable_debug_layers);
  
  // Returns the list of available physical devices.
  inline const vector<shared_ptr<VulkanPhysicalDevice>>& physical_devices() const { return physical_devices_; }
  
  // Returns the list of requested instance layers. This is available here
  // because are usually also specified when creating the logical device. (This
  // has been deprecated, but it is said that some old drivers require it.)
  inline const vector<string>& requested_instance_layers() const {
    return requested_instance_layers_;
  }
  
  // Returns the underlying VkInstance.
  inline VkInstance instance() const { return instance_; }
  
 private:
  // Returns a list of available physical devices that support Vulkan.
  // Initialize() must be successfully called before.
  void ListPhysicalDevices(vector<VkPhysicalDevice>* devices) const;
  
  VkInstance instance_;
  VkDebugReportCallbackEXT debug_callback_;
  vector<string> requested_instance_layers_;
  vector<shared_ptr<VulkanPhysicalDevice>> physical_devices_;
};

// Bundles information about supported capabilities, formats and present modes
// for a given combination of physical device and (display) surface.
struct VulkanSwapChainSupport {
  VkSurfaceCapabilitiesKHR capabilities;
  vector<VkSurfaceFormatKHR> formats;
  vector<VkPresentModeKHR> present_modes;
};

// Wraps a VkPhysicalDevice.
//
// This class can only be created by VulkanInstance. The available instances can
// be retrieved from VulkanInstance::physical_devices().
class VulkanPhysicalDevice {
 friend class VulkanInstance;
 public:
  // Queries for swap chain support, given a specific surface.
  void QuerySwapChainSupport(VkSurfaceKHR surface, VulkanSwapChainSupport* result);
  
  // Returns the device properties which can be used to check the suitability of
  // the device.
  inline const VkPhysicalDeviceProperties& properties() const { return properties_; }
  
  // Returns the device features which can be used to check the suitability of
  // the device.
  inline const VkPhysicalDeviceFeatures& features() const { return features_; }
  
  // Returns the list of device extensions available on this device.
  inline const vector<VkExtensionProperties>& available_extensions() const { return available_extensions_; }
  
  // Returns the list of available queue families, which can be used to check
  // the suitability of the device.
  inline const vector<VkQueueFamilyProperties>& queue_families() const { return queue_families_; }
  
  // Returns the underlying VkPhysicalDevice handle.
  inline VkPhysicalDevice device() const { return device_; }
  
 private:
  VulkanPhysicalDevice(VkPhysicalDevice device);
  
  VkPhysicalDeviceProperties properties_;
  VkPhysicalDeviceFeatures features_;
  vector<VkExtensionProperties> available_extensions_;
  vector<VkQueueFamilyProperties> queue_families_;
  VkPhysicalDevice device_;
};

// Wraps a VkDevice (logical device) and related functionality.
class VulkanDevice {
 public:
  // This does not initialize the device yet. Initialize() must be called for
  // that.
  VulkanDevice();
  
  // Destroys the device.
  ~VulkanDevice();
  
  // Initializes the device, returns true on success.
  //
  // physical_device specifies the physical device for which the logical device
  // shall be created. If -1 is passed in for a queue family index, no such
  // queue will be created. If the same queue family index is passed in for more
  // than one queue family type, only one queue of this family will be created.
  bool Initialize(VkPhysicalDevice physical_device,
                  const vector<string>& extensions,
                  int graphics_queue_family_index,
                  int presentation_queue_family_index,
                  const VulkanInstance& instance);
  
  inline VkQueue graphics_queue() const { return graphics_queue_; }
  inline int graphics_queue_family_index() const { return graphics_queue_family_index_; }
  inline VkQueue presentation_queue() const { return presentation_queue_; }
  inline int presentation_queue_family_index() const { return presentation_queue_family_index_; }
  
  // Returns the underlying VkDevice handle.
  inline VkDevice device() const { return logical_device_; }
  
 private:
  VkDevice logical_device_;
  VkQueue graphics_queue_;
  int graphics_queue_family_index_;
  VkQueue presentation_queue_;
  int presentation_queue_family_index_;
};

}
