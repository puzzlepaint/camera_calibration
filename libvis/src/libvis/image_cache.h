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

#include <map>

#include "libvis/image.h"
#include "libvis/libvis.h"

namespace vis {

template<typename T>
class ImageCache;

// Base class for operations stored in an ImageCache's operation tree.
template<typename T>
class ImageCacheElement {
 public:
  virtual ~ImageCacheElement() {}
  
  inline shared_ptr<ImageCacheElement<T>>* GetOrAllocateElement(const string& key) {
    return &element_map_[key];
  }
  
  virtual void* GetOrComputeResultVoid() = 0;
  
 private:
  // Next level of the operation tree.
  map<string, shared_ptr<ImageCacheElement<T>>> element_map_;
};

// Return value of operation functions for ImageCache.
template<typename T, typename ParentCacheElementType>
struct ImageCacheElementData {
  ImageCacheElementData(ImageCache<T>* image_cache)
      : image_cache(image_cache) {}
  
  ImageCacheElementData(ImageCache<T>* image_cache, ParentCacheElementType* parent_cache_element)
      : image_cache(image_cache), parent_cache_element(parent_cache_element) {}
  
  inline typename ParentCacheElementType::ReturnType GetOrComputeResult() {
    return *reinterpret_cast<typename ParentCacheElementType::ReturnType*>(
        parent_cache_element->GetOrComputeResultVoid());
  }
  
  ImageCache<T>* image_cache;
  ParentCacheElementType* parent_cache_element;
};

// Stores an image and data derived from it such as gradient images and image
// pyramids. Has the ability to derive those on-demand when they are first
// accessed, such that data which is never accessed will not be computed. Can
// even read the original image from disk on demand. Is extensible regarding the
// type of derived data that can be stored.
// 
// Design alternatives for accessing derived data for an example:
// 
// GradientImageX(ImagePyramid(&image_cache, level)).result();
// - ImagePyramid() returns a pair of its name, result, and image_cache pointer.
// - Functions are overloaded such that they can be called with both the initial
//   image_cache, and the intermediate result.
// - result() needs to be called at the end to get the result object (image).
//   The return object can be templated to give the desired return type.
// - Operations allocate worker objects and cache them in the ImageCache, so
//   they are only created once (but this means they should be small). They
//   store the operation (via their virtual function table), a pointer to the
//   previous operation (or root), and their result. The result() call evaluates
//   all operations which don't have a result yet from top to bottom. Worker
//   classes can be templated to know the output type of their predecessor step
//   and be able to call the predecessor operation.
// 
// result = image_cache.Get(ImagePyramid<T>(level), GradientImageX<T>());
// - Get<>() as variadic template, steps have a name (at runtime only so the
//   pyramid can include the level) and a compute operator.
// - Can return different types by returning RightmostArg::ResultType.
template<typename T>
class ImageCache : public ImageCacheElement<T> {
 public:
  typedef shared_ptr<Image<T>> ReturnType;
  
  // Creates an empty image cache. The path and / or image need to be set later.
  inline ImageCache() {}
  
  // Creates an image cache with the given image path. The image is only loaded
  // from disk if it is accessed (or the loading is triggered directly with
  // EnsureImageIsLoaded()).
  inline ImageCache(const string& image_path)
      : image_path_(image_path) {}
  
  // Creates an image cache based on an existing image.
  inline ImageCache(const shared_ptr<Image<T>>& image)
      : image_(image) {}
  
  // Creates an image cache based on an existing image with an image file.
  inline ImageCache(const string& image_path, const shared_ptr<Image<T>>& image)
      : image_path_(image_path), image_(image) {}
  
  inline void SetPath(const string& image_path) {
    image_path_ = image_path;
  }
  
  inline void SetImage(const shared_ptr<Image<T>>& image) {
    image_ = image;
  }
  
  // Tries to read the image from disk if it is not loaded. Returns true if the
  // image is loaded after the function executed, false otherwise.
  bool EnsureImageIsLoaded() {
    if (image_) {
      return true;
    }
    if (image_path_.empty()) {
      return false;
    }
    image_.reset(new Image<T>());
    if (image_->Read(image_path_)) {
      return true;
    }
    image_.reset();
    return false;
  }
  
  // Tries to read the image from disk if it is not loaded. Returns the image
  // shared_ptr or a null shared_ptr if the image could not be loaded.
  inline const shared_ptr<Image<T>>& GetImage() {
    if (!EnsureImageIsLoaded()) {
      // Make sure that image_ is not a valid pointer.
      image_.reset();
    }
    return image_;
  }
  
  // Frees all derived data, but not the original image.
  inline void ClearDerivedData() {
    element_map_.clear();
  }
  
  // Frees the image and all derived data. Only do this if there is a copy of
  // the image on disk given as image path.
  inline void ClearImageAndDerivedData() {
    ClearDerivedData();
    image_.reset();
  }
  
  // Alias for GetImage() which allows the ImageCache to be used as the first
  // operation in a chain.
  inline ReturnType* GetOrComputeResult() {
    EnsureImageIsLoaded();
    return &image_;
  }
  inline virtual void* GetOrComputeResultVoid() override {
    return GetOrComputeResult();
  }
  
  
  inline bool IsImageLoaded() const {
    return image_ ? true : false;
  }
  
  inline const string& image_path() const {
    return image_path_;
  }
  
 private:
  // First level of the operation tree.
  map<string, shared_ptr<ImageCacheElement<T>>> element_map_;
  
  string image_path_;
  shared_ptr<Image<T>> image_;
};



// ### Image pyramid. ###

// Image pyramid cache element.
template<typename T>
class ImagePyramidCacheElement : public ImageCacheElement<T> {
 public:
  typedef shared_ptr<Image<T>> ReturnType;  // TODO: Should this be an ImageCache itself such that additional operations can be applied to the result of the image pyramid downsampling?
  
  ImagePyramidCacheElement(ImageCacheElement<T>* parent_cache_element)
      : parent_cache_element_(parent_cache_element) {}
  
  ReturnType* GetOrComputeResult() {
    // Compute the result only if necessary.
    if (pyramid_image_) {
      return &pyramid_image_;
    }
    
    // We expect a shared_ptr<Image<T>> from the parent. This is unfortunately
    // retrieved in a non-type-safe way since we don't know the type of the
    // parent here, and we want to allow arbitrary result types, so virtual
    // functions do not work.
    shared_ptr<Image<T>>* input =
        reinterpret_cast<shared_ptr<Image<T>>*>(
            parent_cache_element_->GetOrComputeResultVoid());
    
    // Compute the result of this step.
    pyramid_image_.reset(new Image<T>());
    (*input)->DownscaleToHalfSize(pyramid_image_.get());
    
    return &pyramid_image_;
  }
  virtual void* GetOrComputeResultVoid() override {
    return GetOrComputeResult();
  }
  
 private:
  // Previous element in the operation tree.
  ImageCacheElement<T>* parent_cache_element_;
  
  // The result of this operation.
  ReturnType pyramid_image_;
};

// Image pyramid tree constructor function.
template<typename T, typename ParentCacheElementType>
ImageCacheElementData<T, ImagePyramidCacheElement<T>> ImagePyramid(
    const ImageCacheElementData<T, ParentCacheElementType>& parent_cache_element_data,
    u32 pyramid_level) {
  // Get or allocate the next element named "ImagePyramid". Fill in the cache
  // data struct to be used by the next step.
  constexpr const char* name = "ImagePyramid";
  ImageCacheElement<T>* parent_cache_element =
      parent_cache_element_data.parent_cache_element;
  shared_ptr<ImageCacheElement<T>>* element =
      parent_cache_element->GetOrAllocateElement(name);
  
  ImageCacheElementData<T, ImagePyramidCacheElement<T>> cache_data(parent_cache_element_data.image_cache);
  if (*element) {
    // The cache element for this level already exists, convert the pointer to
    // it to the derived type.
    cache_data.parent_cache_element =
        reinterpret_cast<ImagePyramidCacheElement<T>*>(element->get());
  } else {
    // Allocate the cache element for this level.
    cache_data.parent_cache_element = new ImagePyramidCacheElement<T>(parent_cache_element);
    element->reset(cache_data.parent_cache_element);
  }
  
  if (pyramid_level == 1) {
    // End the recursion.
    return cache_data;
  } else {
    return ImagePyramid(cache_data, pyramid_level - 1);
  }
}

// Image pyramid tree root constructor function.
template<typename T>
ImageCacheElementData<T, ImagePyramidCacheElement<T>> ImagePyramid(ImageCache<T>* image_cache, u32 pyramid_level) {
  ImageCacheElementData<T, ImageCache<T>> root_cache_data(image_cache, image_cache);
  return ImagePyramid(root_cache_data, pyramid_level);
}



// ### BilateralFiltered ###

// Cache element.
template<typename T>
class BilateralFilteredCacheElement : public ImageCacheElement<T> {
 public:
  typedef shared_ptr<Image<T>> ReturnType;
  
  BilateralFilteredCacheElement(ImageCacheElement<T>* parent_cache_element, float sigma_xy, const T sigma_value, const T value_to_ignore, float radius_factor)
      : parent_cache_element_(parent_cache_element),
        sigma_xy_(sigma_xy),
        sigma_value_(sigma_value),
        value_to_ignore_(value_to_ignore),
        radius_factor_(radius_factor) {}
  
  ReturnType* GetOrComputeResult() {
    // Compute the result only if necessary.
    if (filtered_image_) {
      return &filtered_image_;
    }
    
    // We expect a shared_ptr<Image<T>> from the parent. This is unfortunately
    // retrieved in a non-type-safe way since we don't know the type of the
    // parent here, and we want to allow arbitrary result types, so virtual
    // functions do not work.
    shared_ptr<Image<T>>* input =
        reinterpret_cast<shared_ptr<Image<T>>*>(
            parent_cache_element_->GetOrComputeResultVoid());
    
    // Compute the result of this step.
    filtered_image_.reset(new Image<T>());
    (*input)->BilateralFilter(sigma_xy_, sigma_value_, value_to_ignore_, radius_factor_, filtered_image_.get());
    
    return &filtered_image_;
  }
  virtual void* GetOrComputeResultVoid() override {
    return GetOrComputeResult();
  }
  
 private:
  // Previous element in the operation tree.
  ImageCacheElement<T>* parent_cache_element_;
  
  // The parameters of this operation.
  float sigma_xy_;
  const T sigma_value_;
  const T value_to_ignore_;
  float radius_factor_;
  
  // The result of this operation.
  ReturnType filtered_image_;
};

// Operator function.
template<typename T>
ImageCacheElementData<T, BilateralFilteredCacheElement<T>> BilateralFiltered(ImageCache<T>* image_cache, float sigma_xy, const T sigma_value, const T value_to_ignore, float radius_factor) {
  ImageCacheElementData<T, BilateralFilteredCacheElement<T>> cache_data(image_cache);
  
  ostringstream name;
  name << "BilateralFiltered_" << sigma_xy << "_" << sigma_value << "_" << value_to_ignore << " " << radius_factor;
  shared_ptr<ImageCacheElement<T>>* element =
      image_cache->GetOrAllocateElement(name.str());
  if (*element) {
    // The cache element for this level already exists, convert the pointer to
    // it to the derived type.
    cache_data.parent_cache_element =
        reinterpret_cast<BilateralFilteredCacheElement<T>*>(element->get());
  } else {
    // Allocate the cache element for this level.
    cache_data.parent_cache_element = new BilateralFilteredCacheElement<T>(image_cache, sigma_xy, sigma_value, value_to_ignore, radius_factor);
    element->reset(cache_data.parent_cache_element);
  }
  
  return cache_data;
}

// Operator function for concatenation.
// TODO: Unify this with the function above as far as possible.
template<typename T, typename ParentCacheElementType>
ImageCacheElementData<T, BilateralFilteredCacheElement<T>> BilateralFiltered(
    const ImageCacheElementData<T, ParentCacheElementType>& parent_cache_element_data,
    float sigma_xy, const T sigma_value, const T value_to_ignore, float radius_factor) {
  ImageCacheElementData<T, BilateralFilteredCacheElement<T>> cache_data(parent_cache_element_data.image_cache);
  
  ostringstream name;
  name << "BilateralFiltered_" << sigma_xy << "_" << sigma_value << "_" << value_to_ignore << " " << radius_factor;
  shared_ptr<ImageCacheElement<T>>* element =
      parent_cache_element_data.parent_cache_element->GetOrAllocateElement(name.str());
  if (*element) {
    // The cache element for this level already exists, convert the pointer to
    // it to the derived type.
    cache_data.parent_cache_element =
        reinterpret_cast<BilateralFilteredCacheElement<T>*>(element->get());
  } else {
    // Allocate the cache element for this level.
    cache_data.parent_cache_element = new BilateralFilteredCacheElement<T>(parent_cache_element_data.parent_cache_element, sigma_xy, sigma_value, value_to_ignore, radius_factor);
    element->reset(cache_data.parent_cache_element);
  }
  
  return cache_data;
}



// ### MaxCutoff ###

// Cache element.
template<typename T>
class MaxCutoffCacheElement : public ImageCacheElement<T> {
 public:
  typedef shared_ptr<Image<T>> ReturnType;
  
  MaxCutoffCacheElement(ImageCacheElement<T>* parent_cache_element, const T max_value, const T replacement_value)
      : parent_cache_element_(parent_cache_element),
        max_value_(max_value),
        replacement_value_(replacement_value) {}
  
  ReturnType* GetOrComputeResult() {
    // Compute the result only if necessary.
    if (filtered_image_) {
      return &filtered_image_;
    }
    
    // We expect a shared_ptr<Image<T>> from the parent. This is unfortunately
    // retrieved in a non-type-safe way since we don't know the type of the
    // parent here, and we want to allow arbitrary result types, so virtual
    // functions do not work.
    shared_ptr<Image<T>>* input =
        reinterpret_cast<shared_ptr<Image<T>>*>(
            parent_cache_element_->GetOrComputeResultVoid());
    
    // Compute the result of this step.
    filtered_image_.reset(new Image<T>());
    (*input)->MaxCutoff(max_value_, replacement_value_, filtered_image_.get());
    
    return &filtered_image_;
  }
  virtual void* GetOrComputeResultVoid() override {
    return GetOrComputeResult();
  }
  
 private:
  // Previous element in the operation tree.
  ImageCacheElement<T>* parent_cache_element_;
  
  // The parameters of this operation.
  const T max_value_;
  const T replacement_value_;
  
  // The result of this operation.
  ReturnType filtered_image_;
};

// Operator function. Replaces all values larger than max_value with replacement_value.
template<typename T>
ImageCacheElementData<T, MaxCutoffCacheElement<T>> MaxCutoff(ImageCache<T>* image_cache, const T max_value, const T replacement_value) {
  ImageCacheElementData<T, MaxCutoffCacheElement<T>> cache_data(image_cache);
  
  ostringstream name;
  name << "MaxCutoff_" << max_value << "_" << replacement_value;
  shared_ptr<ImageCacheElement<T>>* element =
      image_cache->GetOrAllocateElement(name.str());
  if (*element) {
    // The cache element for this level already exists, convert the pointer to
    // it to the derived type.
    cache_data.parent_cache_element =
        reinterpret_cast<MaxCutoffCacheElement<T>*>(element->get());
  } else {
    // Allocate the cache element for this level.
    cache_data.parent_cache_element = new MaxCutoffCacheElement<T>(image_cache, max_value, replacement_value);
    element->reset(cache_data.parent_cache_element);
  }
  
  return cache_data;
}

}
