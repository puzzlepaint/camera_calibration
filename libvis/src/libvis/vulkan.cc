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


#include "libvis/vulkan.h"

#include <set>

#include "libvis/logging.h"

namespace vis {

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
    VkDebugReportFlagsEXT flags,
    VkDebugReportObjectTypeEXT /*obj_type*/,
    uint64_t /*obj*/,
    size_t /*location*/,
    int32_t /*code*/,
    const char* /*layer_prefix*/,
    const char* msg,
    void* /*user_data*/) {
  if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) {
    LOG(ERROR) << "Vulkan debug callback: " << msg << endl;
  } else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) {
    LOG(WARNING) << "Vulkan debug callback: " << msg << endl;
  } else {
    LOG(INFO) << "Vulkan debug callback: " << msg << endl;
  }
  return VK_FALSE;
}

VulkanInstance::VulkanInstance()
    : instance_(nullptr),
      debug_callback_(nullptr) {}

VulkanInstance::~VulkanInstance() {
  if (instance_) {
    if (debug_callback_) {
      auto vkDestroyDebugReportCallbackEXT = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance_, "vkDestroyDebugReportCallbackEXT"));
      if (vkDestroyDebugReportCallbackEXT) {
        vkDestroyDebugReportCallbackEXT(instance_, debug_callback_, nullptr);
      }
    }
    
    vkDestroyInstance(instance_, nullptr);
  }
}

bool VulkanInstance::Initialize(const vector<string>& extensions, bool enable_debug_layers) {
  VLOG(1) << "Initializing Vulkan ...";
  
  // List the available instance layers.
  u32 layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  vector<VkLayerProperties> available_instance_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_instance_layers.data());
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Available instance layers:";
    for (const VkLayerProperties& layer_properties : available_instance_layers) {
      VLOG(1) << "  " << layer_properties.layerName << " (spec version: " << layer_properties.specVersion << ", impl version: " << layer_properties.implementationVersion << ")";
    }
  }
  
  // List the available instance extensions.
  // NOTE: Not filtering the extensions for a specific layer (1st parameter).
  u32 extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  vector<VkExtensionProperties> available_instance_extensions(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_instance_extensions.data());
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Available instance extensions:";
    for (const VkExtensionProperties& extension_properties : available_instance_extensions) {
      VLOG(1) << "  " << extension_properties.extensionName << " (spec version: " << extension_properties.specVersion << ")";
    }
  }
  
  // Select the layers and extensions to use.
  vector<string> requested_instance_extensions = extensions;
  if (enable_debug_layers) {
    requested_instance_layers_.push_back("VK_LAYER_LUNARG_standard_validation");
    requested_instance_extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
  }
  
  // Check whether the requested layers and extensions are supported.
  for (const string& requested_layer : requested_instance_layers_) {
    bool found = false;
    for (const VkLayerProperties& layer_properties : available_instance_layers) {
      if (requested_layer == layer_properties.layerName) {
        found = true;
        break;
      }
    }
    if (!found) {
      LOG(ERROR) << "Requested instance layer is not supported: " << requested_layer;
      return false;
    }
  }
  
  for (const string& requested_extension : requested_instance_extensions) {
    bool found = false;
    for (const VkExtensionProperties& extension_properties : available_instance_extensions) {
      if (requested_extension == extension_properties.extensionName) {
        found = true;
        break;
      }
    }
    if (!found) {
      LOG(ERROR) << "Requested instance extension is not supported: " << requested_extension;
      return false;
    }
  }
  
  // Specify application information (optional).
  VkApplicationInfo app_info = {};
  app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext              = nullptr;
  app_info.pApplicationName   = "libvis";
  app_info.applicationVersion = 0;
  app_info.pEngineName        = "libvis";
  app_info.engineVersion      = 0;
  app_info.apiVersion         = VK_API_VERSION_1_0;
  
  // Create VkInstance.
  VkInstanceCreateInfo instance_info = {};
  instance_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pNext                   = nullptr;
  instance_info.flags                   = 0;
  instance_info.pApplicationInfo        = &app_info;
  instance_info.enabledLayerCount       = requested_instance_layers_.size();
  vector<const char*> layer_name_pointers(requested_instance_layers_.size());
  for (usize i = 0; i < requested_instance_layers_.size(); ++ i) {
    layer_name_pointers[i] = requested_instance_layers_[i].data();
  }
  instance_info.ppEnabledLayerNames     = layer_name_pointers.data();
  instance_info.enabledExtensionCount   = requested_instance_extensions.size();
  vector<const char*> extension_name_pointers(requested_instance_extensions.size());
  for (usize i = 0; i < requested_instance_extensions.size(); ++ i) {
    extension_name_pointers[i] = requested_instance_extensions[i].data();
  }
  instance_info.ppEnabledExtensionNames = extension_name_pointers.data();
  VkResult result = vkCreateInstance(&instance_info, nullptr, &instance_);
  if (result != VK_SUCCESS) {
    LOG(ERROR) << "vkCreateInstance() failed with result: " << result;
    return false;
  }
  
  // Setup debug callback if requested.
  if (enable_debug_layers) {
    VkDebugReportCallbackCreateInfoEXT debug_report_info = {};
    debug_report_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    debug_report_info.flags =
        VK_DEBUG_REPORT_WARNING_BIT_EXT |
        VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT |
        VK_DEBUG_REPORT_ERROR_BIT_EXT;
    // NOTE: Additional flag values are:
    //       VK_DEBUG_REPORT_INFORMATION_BIT_EXT
    //       VK_DEBUG_REPORT_DEBUG_BIT_EXT
    debug_report_info.pfnCallback = &VulkanDebugCallback;
    
    auto vkCreateDebugReportCallbackEXT = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance_, "vkCreateDebugReportCallbackEXT"));
    if (vkCreateDebugReportCallbackEXT) {
      VkResult result = vkCreateDebugReportCallbackEXT(instance_, &debug_report_info, nullptr, &debug_callback_);
      if (result != VK_SUCCESS) {
        LOG(ERROR) << "vkCreateDebugReportCallbackEXT() failed with result: " << result;
        return false;
      }
    } else {
      LOG(ERROR) << "vkCreateDebugReportCallbackEXT() not available.";
      return false;
    }
  }
  
  // List the physical devices.
  vector<VkPhysicalDevice> devices;
  ListPhysicalDevices(&devices);
  physical_devices_.reserve(devices.size());
  for (VkPhysicalDevice device : devices) {
    physical_devices_.push_back(shared_ptr<VulkanPhysicalDevice>(new VulkanPhysicalDevice(device)));
  }
  
  return true;
}

void VulkanInstance::ListPhysicalDevices(vector<VkPhysicalDevice>* devices) const {
  u32 device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  devices->resize(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices->data());
}


void VulkanPhysicalDevice::QuerySwapChainSupport(VkSurfaceKHR surface, VulkanSwapChainSupport* result) {
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_, surface, &result->capabilities);
  
  u32 format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, &format_count, nullptr);
  if (format_count != 0) {
    result->formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device_, surface, &format_count, result->formats.data());
  }
  
  u32 present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device_, surface, &present_mode_count, nullptr);
  if (present_mode_count != 0) {
    result->present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device_, surface, &present_mode_count, result->present_modes.data());
  }
}

VulkanPhysicalDevice::VulkanPhysicalDevice(VkPhysicalDevice device)
    : device_(device) {
  vkGetPhysicalDeviceProperties(device, &properties_);
  vkGetPhysicalDeviceFeatures(device, &features_);
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "Vulkan device:";
    VLOG(1) << "  Name: " << properties_.deviceName;
    VLOG(1) << "  ID: " << properties_.deviceID;
    string type_string;
    if (properties_.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER) {
      type_string = "other";
    } else if (properties_.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      type_string = "integrated GPU";
    } else if (properties_.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      type_string = "discrete GPU";
    } else if (properties_.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
      type_string = "virtual GPU";
    } else if (properties_.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
      type_string = "CPU";
    } else {
      type_string = "unknown";
    }
    VLOG(1) << "  Type: " << type_string;
    VLOG(1) << "  API version: " << properties_.apiVersion;
    VLOG(1) << "  Driver version: " << properties_.driverVersion;
    VLOG(1) << "  Vendor ID: " << properties_.vendorID;
    // NOTE: Properties which are not logged:
    //       pipelineCacheUUID
    //       limits
    //       sparseProperties
    // NOTE: Not logging features.
  }
  
  u32 queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
  queue_families_.resize(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families_.data());
  if (VLOG_IS_ON(1)) {
    for (u32 queue_family_index = 0; queue_family_index < queue_family_count; ++ queue_family_index) {
      const VkQueueFamilyProperties& family_properties = queue_families_[queue_family_index];
      VLOG(1) << "  Queue family:";
      std::string queue_flags_string;
      if (family_properties.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        if (!queue_flags_string.empty()) {
          queue_flags_string += " | ";
        }
        queue_flags_string += "graphics";
      }
      if (family_properties.queueFlags & VK_QUEUE_COMPUTE_BIT) {
        if (!queue_flags_string.empty()) {
          queue_flags_string += " | ";
        }
        queue_flags_string += "compute";
      }
      if (family_properties.queueFlags & VK_QUEUE_TRANSFER_BIT) {
        if (!queue_flags_string.empty()) {
          queue_flags_string += " | ";
        }
        queue_flags_string += "transfer";
      }
      if (family_properties.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) {
        if (!queue_flags_string.empty()) {
          queue_flags_string += " | ";
        }
        queue_flags_string += "sparse binding";
      }
      VLOG(1) << "    Queue flags: " << queue_flags_string;
      VLOG(1) << "    Queue count: " << family_properties.queueCount;
      VLOG(1) << "    Timestamp valid bits: " << family_properties.timestampValidBits;
      VLOG(1) << "    Min image transfer granularity (width, height, depth): ("
                << family_properties.minImageTransferGranularity.width << ", "
                << family_properties.minImageTransferGranularity.height << ", "
                << family_properties.minImageTransferGranularity.depth << ")";
    }
  }
  
  u32 extension_count;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
  available_extensions_.resize(extension_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions_.data());
}


VulkanDevice::VulkanDevice()
    : logical_device_(nullptr),
      graphics_queue_(nullptr),
      presentation_queue_(nullptr) {}

VulkanDevice::~VulkanDevice() {
  if (logical_device_) {
    vkDestroyDevice(logical_device_, nullptr);
  }
}

bool VulkanDevice::Initialize(
    VkPhysicalDevice physical_device,
    const vector<string>& extensions,
    int graphics_queue_family_index,
    int presentation_queue_family_index,
    const VulkanInstance& instance) {
  graphics_queue_family_index_ = graphics_queue_family_index;
  presentation_queue_family_index_ = presentation_queue_family_index;
  
  set<int> queue_family_indices;
  queue_family_indices.insert(graphics_queue_family_index);
  queue_family_indices.insert(presentation_queue_family_index);
  
  float queue_priority = 1.0f;
  vector<VkDeviceQueueCreateInfo> queue_create_infos;
  for (int queue_family_index : queue_family_indices) {
    if (queue_family_index < 0) {
      continue;
    }
    
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.pNext = nullptr;
    queue_info.flags = 0;
    queue_info.queueFamilyIndex = queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_info);
  }

  VkPhysicalDeviceFeatures device_features = {};
  
  VkDeviceCreateInfo device_info = {};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.pNext = nullptr;
  device_info.flags = 0;
  device_info.queueCreateInfoCount = queue_create_infos.size();
  device_info.pQueueCreateInfos = queue_create_infos.data();
  device_info.enabledLayerCount = instance.requested_instance_layers().size();
  vector<const char*> layer_name_pointers(instance.requested_instance_layers().size());
  for (usize i = 0; i < instance.requested_instance_layers().size(); ++ i) {
    layer_name_pointers[i] = instance.requested_instance_layers()[i].data();
  }
  device_info.ppEnabledLayerNames = layer_name_pointers.data();
  device_info.enabledExtensionCount = extensions.size();
  vector<const char*> extension_name_pointers(extensions.size());
  for (usize i = 0; i < extensions.size(); ++ i) {
    extension_name_pointers[i] = extensions[i].data();
  }
  device_info.ppEnabledExtensionNames = extension_name_pointers.data();
  device_info.pEnabledFeatures = &device_features;
  
  if (vkCreateDevice(physical_device, &device_info, nullptr, &logical_device_) != VK_SUCCESS) {
    LOG(ERROR) << "Failed to create logical device!";
    return false;
  }
  
  // Get the handles to the created queues.
  if (graphics_queue_family_index < 0) {
    graphics_queue_ = nullptr;
  } else {
    vkGetDeviceQueue(logical_device_, graphics_queue_family_index, 0, &graphics_queue_);
  }
  
  if (presentation_queue_family_index < 0) {
    presentation_queue_ = nullptr;
  } else {
    vkGetDeviceQueue(logical_device_, presentation_queue_family_index, 0, &presentation_queue_);
  }
  
  return true;
}

}
