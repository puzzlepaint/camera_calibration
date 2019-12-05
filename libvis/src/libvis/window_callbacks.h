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

#include <QInputEvent>

// QT_BEGIN_NAMESPACE
class QPainter;
// QT_END_NAMESPACE

namespace vis {

class RenderWindow;
class ImageDisplayQtWindow;

// Interface which can be subclassed to receive render window callbacks.
// Attention: these callbacks are called from the Qt thread (if Qt is used as
// window toolkit), and therefore likely from a different thread than the main
// thread.
template <class RenderWindowT>
class WindowCallbacks {
 public:
  enum class MouseButton {
    kLeft   = 1 << 0,
    kMiddle = 1 << 1,
    kRight  = 1 << 2
  };
  
  enum class Modifier {
    kShift = 1 << 0,
    kCtrl  = 1 << 1,
    kAlt   = 1 << 2
  };
  
  virtual ~WindowCallbacks() = default;
  
  // Called as the first callback to notify this class about the RenderWindow it
  // is associated with.
  void SetRenderWindow(RenderWindowT* window) {
    window_ = window;
  }
  
  // Called when the render window is initialized. Can be used to allocate
  // rendering API resources that require a rendering API context.
  virtual void Initialize() {}
  
  // Called when the render window is destroyed. Can be used to deallocate
  // rendering API resources that require a rendering API context.
  virtual void Deinitialize() {}
  
  // Called after the window is resized. Can be used to re-allocate rendering
  // API resources in the right size.
  virtual void Resize(int width, int height) {
    (void) width;
    (void) height;
  }
  
  // Called when the scene shall be rendered (with a graphics API such as OpenGL or Vulkan).
  virtual void Render() {}
  
  // Called when the scene shall be rendered (on top of the displayed image, with the QPainter API).
  // TODO: Do not require Qt includes to be present to use this class! Let the user use the API via a generic image display or window instead (the QPainter for Qt could be gotten from the render window that is set via SetRenderWindow() in the beginning).
  virtual void Render(QPainter* /*painter*/) {}
  
  // Called when the user presses a mouse button.
  virtual void MouseDown(MouseButton button, int x, int y) {
    (void) button;
    (void) x;
    (void) y;
  }
  
  // Called when the user moves the mouse over the window (both if a mouse
  // button is pressed and if not).
  virtual void MouseMove(int x, int y) {
    (void) x;
    (void) y;
  }
  
  // Called when the user releases a mouse button.
  virtual void MouseUp(MouseButton button, int x, int y) {
    (void) button;
    (void) x;
    (void) y;
  }
  
  // Called when the mouse wheel is rotated.
  virtual void WheelRotated(float degrees, Modifier modifiers) {
    (void) degrees;
    (void) modifiers;
  }
  
  // Called when a key is pressed.
  virtual void KeyPressed(char key, Modifier modifiers) {
    (void) key;
    (void) modifiers;
  }
  
  // Called when a key is released.
  virtual void KeyReleased(char key, Modifier modifiers) {
    (void) key;
    (void) modifiers;
  }
  
  // TODO: Separate functions for touches (since they are different and could
  //       be mapped to mouse button presses, but don't need to be).
  
  // TODO: Should be in a separate Qt-specific file
  static Modifier ConvertQtModifiers(QInputEvent* event) {
    int result = 0;
    if (event->modifiers() & Qt::ShiftModifier) {
      result |= static_cast<int>(Modifier::kShift);
    }
    if (event->modifiers() & Qt::ControlModifier) {
      result |= static_cast<int>(Modifier::kCtrl);
    }
    if (event->modifiers() & Qt::AltModifier) {
      result |= static_cast<int>(Modifier::kAlt);
    }
    return static_cast<Modifier>(result);
  }

 protected:
  // Not owned.
  RenderWindowT* window_;
};

typedef WindowCallbacks<RenderWindow> RenderWindowCallbacks;
typedef WindowCallbacks<ImageDisplayQtWindow> ImageWindowCallbacks;

}
