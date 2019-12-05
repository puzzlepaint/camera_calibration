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


#include "libvis/image_display_qt_widget.h"

#include <cmath>

#include <QPainter>
#include <QPaintEvent>

#include "libvis/image_display_qt_window.h"

namespace vis {

ImageDisplayQtWidget::ImageDisplayQtWidget(ImageDisplayQtWindow* window, QWidget* parent)
    : QWidget(parent),
      callbacks_(nullptr),
      window_(window) {
  dragging_ = false;
  
  view_scale_ = 1.0;
  view_offset_x_ = 0.0;
  view_offset_y_ = 0.0;
  qimage_ = QImage();
  UpdateViewTransforms();
  
  setAttribute(Qt::WA_OpaquePaintEvent);
  setAutoFillBackground(false);
  setMouseTracking(true);
  setFocusPolicy(Qt::ClickFocus);
  setSizePolicy(QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred));
}

ImageDisplayQtWidget::~ImageDisplayQtWidget() {}

void ImageDisplayQtWidget::SetCallbacks(const shared_ptr<ImageWindowCallbacks>& callbacks) {
  callbacks_ = callbacks;
  update(rect());
}

void ImageDisplayQtWidget::SetViewOffset(double x, double y) {
  window_->FitContent(false);
  view_offset_x_ = x;
  view_offset_y_ = y;
  UpdateViewTransforms();
  update(rect());
}

void ImageDisplayQtWidget::SetZoomFactor(double zoom_factor) {
  window_->FitContent(false);
  view_scale_ = zoom_factor;
  emit ZoomChanged(view_scale_);
  UpdateViewTransforms();
  update(rect());
}

void ImageDisplayQtWidget::ZoomAt(int x, int y, double target_zoom) {
  // viewport_to_image_.m11() * pos.x() + viewport_to_image_.m13() == (pos.x() - (new_view_offset_x_ + (0.5 * width()) - (0.5 * image_.width()) * new_view_scale_)) / new_view_scale_;
  QPointF center_on_image = ViewportToImage(QPoint(x, y));
  view_offset_x_ = x - (0.5 * width() - (0.5 * qimage_.width()) * target_zoom) - target_zoom * center_on_image.x();
  view_offset_y_ = y - (0.5 * height() - (0.5 * qimage_.height()) * target_zoom) - target_zoom * center_on_image.y();
  SetZoomFactor(target_zoom);
}

void ImageDisplayQtWidget::FitContent(bool update_display) {
  if (qimage_.isNull()) {
    return;
  }
  
  // Center image
  view_offset_x_ = 0.0;
  view_offset_y_ = 0.0;
  
  // Scale image such that it exactly fills the widget
  view_scale_ = std::min(width() / (1. * qimage_.width()),
                         height() / (1. * qimage_.height()));
  emit ZoomChanged(view_scale_);
  
  if (update_display) {
    UpdateViewTransforms();
    update(rect());
  }
}

void ImageDisplayQtWidget::AddSubpixelDotPixelCornerConv(float x, float y, u8 r, u8 g, u8 b) {
  dots_.emplace_back();
  SubpixelDot* new_dot = &dots_.back();
  new_dot->xy = QPointF(x, y);
  new_dot->rgb = qRgb(r, g, b);
}

void ImageDisplayQtWidget::AddSubpixelLinePixelCornerConv(float x0, float y0, float x1, float y1, u8 r, u8 g, u8 b) {
  lines_.emplace_back();
  SubpixelLine* new_line = &lines_.back();
  new_line->xy0 = QPointF(x0, y0);
  new_line->xy1 = QPointF(x1, y1);
  new_line->rgb = qRgb(r, g, b);
}

void ImageDisplayQtWidget::AddSubpixelTextPixelCornerConv(float x, float y, u8 r, u8 g, u8 b, const string& text) {
  texts_.emplace_back();
  SubpixelText* new_text = &texts_.back();
  new_text->xy = QPointF(x, y);
  new_text->rgb = qRgb(r, g, b);
  new_text->text = QString::fromStdString(text);
}

void ImageDisplayQtWidget::Clear() {
  dots_.clear();
  lines_.clear();
  texts_.clear();
}

QSize ImageDisplayQtWidget::sizeHint() const {
  if (qimage_.isNull()) {
    return QSize(150, 150);
  } else {
    return qimage_.size();
  }
}

void ImageDisplayQtWidget::resizeEvent(QResizeEvent* event) {
  if (window_->fit_contents_active()) {
    FitContent(false);
  }
  UpdateViewTransforms();
  QWidget::resizeEvent(event);
  if (callbacks_) {
    callbacks_->Resize(event->size().width(), event->size().height());
  }
}

void ImageDisplayQtWidget::paintEvent(QPaintEvent* event) {
  // Create painter and set its options.
  QPainter painter(this);
  QRect event_rect = event->rect();
  painter.setClipRect(event_rect);
  
  painter.fillRect(event_rect, QColor(Qt::gray));
  
  if (qimage_.isNull()) {
    return;
  }
  
  painter.setRenderHint(QPainter::Antialiasing, true);
  
  QTransform image_to_viewport_T = image_to_viewport_.transposed();
  painter.setTransform(image_to_viewport_T);
  
  painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
  painter.drawImage(QPointF(0, 0), qimage_);
  
  painter.resetTransform();
  
  if (!dots_.empty()) {
    painter.setBrush(Qt::NoBrush);
    
    for (const SubpixelDot& dot : dots_) {
      QPointF viewport_position = image_to_viewport_T.map(dot.xy);
      
      painter.setPen(dot.rgb);
      painter.drawEllipse(viewport_position, 2, 2);
    }
  }
  
  if (!lines_.empty()) {
    painter.setBrush(Qt::NoBrush);
    
    for (const SubpixelLine& line : lines_) {
      QPointF viewport_position_0 = image_to_viewport_T.map(line.xy0);
      QPointF viewport_position_1 = image_to_viewport_T.map(line.xy1);
      
      painter.setPen(line.rgb);
      painter.drawLine(viewport_position_0, viewport_position_1);
    }
  }
  
  if (!texts_.empty()) {
    painter.setBrush(Qt::NoBrush);
    
    for (const SubpixelText& text : texts_) {
      QPointF viewport_position = image_to_viewport_T.map(text.xy);
      
      painter.setPen(text.rgb);
      painter.drawText(viewport_position, text.text);
    }
  }
  
  if (callbacks_) {
    callbacks_->Render(&painter);
  }
  
  painter.end();
}

void ImageDisplayQtWidget::mousePressEvent(QMouseEvent* event) {
  if (dragging_) {
    event->accept();
    return;
  }
  
  ImageWindowCallbacks::MouseButton button;
  if (event->button() == Qt::LeftButton) {
    button = ImageWindowCallbacks::MouseButton::kLeft;
  } else if (event->button() == Qt::MiddleButton) {
    button = ImageWindowCallbacks::MouseButton::kMiddle;
  } else if (event->button() == Qt::RightButton) {
    button = ImageWindowCallbacks::MouseButton::kRight;
  } else {
    return;
  }
  
  if (callbacks_) {
    callbacks_->MouseDown(button, event->pos().x(), event->pos().y());
  }

  if (event->button() == Qt::MiddleButton) {
    startDragging(event->pos());
    event->accept();
  }
}

template <typename T>
void DisplayValue(T value, std::ostringstream* o) {
  *o << value;
}

void DisplayValue(char value, std::ostringstream* o) {
  *o << static_cast<int>(value);
}

void DisplayValue(unsigned char value, std::ostringstream* o) {
  *o << static_cast<int>(value);
}

template <typename Derived>
void DisplayValue(const Eigen::MatrixBase<Derived>& value, std::ostringstream* o) {
  *o << value.transpose();
}

void DisplayValue(const Vec3u8& value, std::ostringstream* o) {
  *o << value.transpose().cast<int>();
}

void DisplayValue(const Vec4u8& value, std::ostringstream* o) {
  *o << value.transpose().cast<int>();
}

void ImageDisplayQtWidget::mouseMoveEvent(QMouseEvent* event) {
  QPointF image_pos = ViewportToImage(event->localPos());
  string pixel_value;
  QRgb pixel_displayed_value = qRgb(0, 0, 0);
  bool pixel_value_valid =
      !qimage_.isNull() && image_pos.x() >= 0 && image_pos.y() >= 0 &&
      image_pos.x() < qimage_.width() && image_pos.y() < qimage_.height();
  if (pixel_value_valid) {
    int x = image_pos.x();
    int y = image_pos.y();
    
    std::ostringstream o;
    // NOTE: Special cases of DisplayValue() for char types are needed
    //       since the values are printed as letters otherwise.
    IDENTIFY_IMAGE(image_, DisplayValue(_image_->at(x, y), &o));
    pixel_value = o.str();
    pixel_displayed_value = qimage_.pixel(x, y);
  }
  emit CursorPositionChanged(image_pos, pixel_value_valid, pixel_value, pixel_displayed_value);
  
  if (dragging_) {
    updateDragging(event->pos());
    return;
  }
  
  if (callbacks_) {
    callbacks_->MouseMove(event->pos().x(), event->pos().y());
  }
}

void ImageDisplayQtWidget::mouseReleaseEvent(QMouseEvent* event) {
  if (dragging_) {
    finishDragging(event->pos());
    event->accept();
    return;
  }
  
  ImageWindowCallbacks::MouseButton button;
  if (event->button() == Qt::LeftButton) {
    button = ImageWindowCallbacks::MouseButton::kLeft;
  } else if (event->button() == Qt::MiddleButton) {
    button = ImageWindowCallbacks::MouseButton::kMiddle;
  } else if (event->button() == Qt::RightButton) {
    button = ImageWindowCallbacks::MouseButton::kRight;
  } else {
    return;
  }
  
  if (callbacks_) {
    callbacks_->MouseUp(button, event->pos().x(), event->pos().y());
  }
  event->accept();
}

void ImageDisplayQtWidget::wheelEvent(QWheelEvent* event) {
  if (event->orientation() == Qt::Vertical) {
    double degrees = event->delta() / 8.0;
    double num_steps = degrees / 15.0;
    
    double scale_factor = pow(sqrt(2.0), num_steps);
    double target_scale = view_scale_ * scale_factor;
    
    ZoomAt(event->pos().x(), event->pos().y(), target_scale);
    
    if (callbacks_ && event->orientation() == Qt::Vertical) {
      callbacks_->WheelRotated(event->delta() / 8.0f, ImageWindowCallbacks::ConvertQtModifiers(event));
    }
  } else {
    event->ignore();
  }
}

void ImageDisplayQtWidget::keyPressEvent(QKeyEvent* event) {
  if (callbacks_ && event->text().size() > 0) {
    callbacks_->KeyPressed(event->text()[0].toLatin1(), ImageWindowCallbacks::ConvertQtModifiers(event));
  }
}

void ImageDisplayQtWidget::keyReleaseEvent(QKeyEvent* event) {
  if (callbacks_ && event->text().size() > 0) {
    callbacks_->KeyReleased(event->text()[0].toLatin1(), ImageWindowCallbacks::ConvertQtModifiers(event));
  }
}

// TODO: Is this useful anywhere? If not, remove.
// template <typename DestT, typename SrcT>
// DestT ScalarOrMatrixCast(const SrcT& value) {
//   return static_cast<DestT>(value);
// }
// 
// template <typename DestT, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
// Eigen::Matrix<DestT, _Rows, _Cols, _Options, _MaxRows, _MaxCols>
//     ScalarOrMatrixCast(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& value) {
//   return value.template cast<DestT>();
// }

// TODO: Could it be that it would be much better readable to simply split
//       the input into scalar and matrix types in AdjustBrightnessContrastHelper()
//       to avoid the need for get_display_type and AdjustBrightnessContrast()?
//       It would duplicate the logic in AdjustBrightnessContrastHelper() though.
template<typename T>
struct get_display_type {
  typedef Vec3u8 Type;
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct get_display_type<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  typedef Matrix<u8, _Rows, _Cols, _Options, _MaxRows, _MaxCols> Type;
};

// HACK: We identify scalar vs. matrix input by defining a type as bool in the
//       scalar case, and as int in the matrix case (types chosen arbitrarily).
template<typename T>
struct get_matrix_identifier_type {
  typedef bool Type;
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct get_matrix_identifier_type<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> {
  typedef int Type;
};

template <typename Identifier, typename SrcT>
void AdjustBrightnessContrast(
    Vec3u8* /*dest*/,
    const SrcT* /*src*/,
    double /*scale*/,
    double /*bias*/,
    Identifier /*dummy*/) {
  LOG(FATAL) << "This should never be called. One of the other overloads should be used instead.";
}

template <typename SrcT>
void AdjustBrightnessContrast(
    Vec3u8* dest,
    const SrcT* src,
    double scale,
    double bias,
    bool /*dummy*/) {
  *dest = Vec3u8::Constant(std::max<double>(0, std::min<double>(255, scale * (*src) + bias)));
}

template <>
void AdjustBrightnessContrast(
    Vec3u8* dest,
    const float* src,
    double scale,
    double bias,
    bool /*dummy*/) {
  if (std::isnan(*src)) {
    *dest = Vec3u8(255, 0, 0);
  } else {
    *dest = Vec3u8::Constant(std::max<double>(0, std::min<double>(255, scale * (*src) + bias)));
  }
}

template <>
void AdjustBrightnessContrast(
    Vec3u8* dest,
    const double* src,
    double scale,
    double bias,
    bool /*dummy*/) {
  if (std::isnan(*src)) {
    *dest = Vec3u8(255, 0, 0);
  } else {
    *dest = Vec3u8::Constant(std::max<double>(0, std::min<double>(255, scale * (*src) + bias)));
  }
}

template <typename _SrcScalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void AdjustBrightnessContrast(
  Vec3u8* dest,
  const Eigen::Matrix<_SrcScalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>* src,
  double scale,
  double bias,
  int /*dummy*/) {
  typedef Eigen::Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols> DoubleMatrixType;
  *dest = DoubleMatrixType::Zero().cwiseMax(
              DoubleMatrixType::Constant(255).cwiseMin(
                  scale * src->template cast<double>() + DoubleMatrixType::Constant(bias))).template cast</*_DestScalar*/ u8>();
}

// TODO: Rename function
template <typename T>
void AdjustBrightnessContrastHelper(Image<T>* image, double white_value, double black_value, QImage* qimage) {
  typedef typename get_display_type<T>::Type DisplayType;  // Will be Vec3u8 for scalar values, or an u8 vector for vectors.
  typedef typename get_matrix_identifier_type<T>::Type IdentifierType;
  Image<DisplayType> display_image(image->width(), image->height());
  double scale = 255.999 / (white_value - black_value);
  double bias = (-255.999 * black_value) / (white_value - black_value);
  for (u32 y = 0; y < image->height(); ++ y) {
    const T* read_ptr = image->row(y);
    DisplayType* write_ptr = display_image.row(y);
    DisplayType* write_end = write_ptr + image->width();
    while (write_ptr < write_end) {
      AdjustBrightnessContrast(write_ptr, read_ptr, scale, bias, IdentifierType());
      ++ write_ptr;
      ++ read_ptr;
    }
  }
  *qimage = display_image.WrapInQImage().copy();
}

template <>
void AdjustBrightnessContrastHelper(Image<Vec2u8>* /*image*/, double /*white_value*/, double /*black_value*/, QImage* /*qimage*/) {
  // Cannot display Vec2u8 images.
  // NOTE: It would be an option to pad them with the 3rd vector component being zero.
  LOG(FATAL) << "Displaying Vec2u8 images is not supported.";
}

template <>
void AdjustBrightnessContrastHelper(Image<Vec4u8>* /*image*/, double /*white_value*/, double /*black_value*/, QImage* /*qimage*/) {
  // Cannot display Vec4u8 images.
  LOG(FATAL) << "Displaying Vec4u8 images is not supported.";
}

void ImageDisplayQtWidget::UpdateQImage() {
  double black_value = window_->black_value_;
  double white_value = window_->white_value_;
  
  // Direct wrapping is only possible if no transformations are applied.
  if (black_value == 0 && white_value == 255) {
    // No brightness / contrast adjustment required.
    if (image_.type() == ImageType::U8) {
      qimage_ = image_.get<u8>()->WrapInQImage();
    } else if (image_.type() == ImageType::Vec3u8) {
      qimage_ = image_.get<Vec3u8>()->WrapInQImage();
    } else if (image_.type() == ImageType::Vec4u8) {
      qimage_ = image_.get<Vec4u8>()->WrapInQImage();
    } else {
      // Try to convert the image to a displayable type using AdjustBrightnessContrastHelper().
      IDENTIFY_IMAGE(image_, AdjustBrightnessContrastHelper(_image_, white_value, black_value, &qimage_));
    }
  } else {
    // Perform brightness / contrast adjustment.
    IDENTIFY_IMAGE(image_, AdjustBrightnessContrastHelper(_image_, white_value, black_value, &qimage_));
  }
  
  if (window_->fit_contents_active()) {
    FitContent(false);
  }
}

void ImageDisplayQtWidget::UpdateViewTransforms() {
  image_to_viewport_.setMatrix(
      view_scale_,           0,   view_offset_x_ + (0.5 * width()) - (0.5 * qimage_.width()) * view_scale_,
                0, view_scale_, view_offset_y_ + (0.5 * height()) - (0.5 * qimage_.height()) * view_scale_,
                0,           0,                                                                         1);
  viewport_to_image_ = image_to_viewport_.inverted();
}

QPointF ImageDisplayQtWidget::ViewportToImage(const QPointF& pos) {
  return QPointF(viewport_to_image_.m11() * pos.x() + viewport_to_image_.m12() * pos.y() + viewport_to_image_.m13(),
                 viewport_to_image_.m21() * pos.x() + viewport_to_image_.m22() * pos.y() + viewport_to_image_.m23());
}

QPointF ImageDisplayQtWidget::ImageToViewport(const QPointF& pos) {
  return QPointF(image_to_viewport_.m11() * pos.x() + image_to_viewport_.m12() * pos.y() + image_to_viewport_.m13(),
                 image_to_viewport_.m21() * pos.x() + image_to_viewport_.m22() * pos.y() + image_to_viewport_.m23());
}

void ImageDisplayQtWidget::startDragging(QPoint pos) {
//   Q_ASSERT(!dragging);
  dragging_ = true;
  drag_start_pos_ = pos;
  normal_cursor_  = cursor();
  setCursor(Qt::ClosedHandCursor);
}

void ImageDisplayQtWidget::updateDragging(QPoint pos) {
//   Q_ASSERT(dragging);
  
  QPoint offset = pos - drag_start_pos_;
  SetViewOffset(view_offset_x_ + offset.x(),
                view_offset_y_ + offset.y());
  
  drag_start_pos_ = pos;
}

void ImageDisplayQtWidget::finishDragging(QPoint pos) {
  updateDragging(pos);
  
  dragging_ = false;
  setCursor(normal_cursor_);
}
}
