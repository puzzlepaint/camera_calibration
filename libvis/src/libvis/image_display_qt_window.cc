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


#include "libvis/image_display_qt_window.h"

#include <mutex>

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QCloseEvent>
#include <QDesktopWidget>
#include <QFileDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QStatusBar>
#include <QToolBar>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidgetAction>

#include "libvis/image_display.h"

#include <iostream>

// This must be done outside of any namespace according to the Qt documentation.
void InitQtResources() {
  static std::mutex init_mutex;
  std::unique_lock<std::mutex> lock(init_mutex);
  static bool resources_initialized = false;
  if (!resources_initialized) {
    // NOTE: I am not sure where this difference in naming came from. Making it
    //       dependent on this Qt version is just a guess so far.
// #if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
    Q_INIT_RESOURCE(resources);
// #else
//     Q_INIT_RESOURCE(libvis_resources_resources);
// #endif
    resources_initialized = true;
  }
}

namespace vis {

class DisplaySettingsAction: public QWidgetAction {
 Q_OBJECT
 public:
  explicit DisplaySettingsAction(ImageDisplayQtWindow* window, QWidget* parent)
      : QWidgetAction(parent), window_(window) {}

 public slots:
  void SetBlackValue() {
    bool ok = false;
    double value = black_value_edit_->text().toDouble(&ok);
    if (ok) {
      window_->black_value_ = value;
      window_->widget().UpdateQImage();
      window_->widget().update(window_->widget().rect());
    }
  }
  
  void SetWhiteValue() {
    bool ok = false;
    double value = white_value_edit_->text().toDouble(&ok);
    if (ok) {
      window_->white_value_ = value;
      window_->widget().UpdateQImage();
      window_->widget().update(window_->widget().rect());
    }
  }
  
 protected:
  QWidget* createWidget(QWidget* parent) {
    QGridLayout* layout = new QGridLayout();
    
    layout->addWidget(new QLabel("Intensity display range min:"), 0, 0);
    
    black_value_edit_ = new QLineEdit(QString::number(window_->black_value_));
    connect(black_value_edit_, SIGNAL(editingFinished()), this, SLOT(SetBlackValue()));
    connect(black_value_edit_, SIGNAL(returnPressed()), this, SLOT(SetBlackValue()));
    layout->addWidget(black_value_edit_, 0, 1);
    
    layout->addWidget(new QLabel("Intensity display range max:"), 1, 0);
    
    white_value_edit_ = new QLineEdit(QString::number(window_->white_value_));
    connect(white_value_edit_, SIGNAL(editingFinished()), this, SLOT(SetWhiteValue()));
    connect(white_value_edit_, SIGNAL(returnPressed()), this, SLOT(SetWhiteValue()));
    layout->addWidget(white_value_edit_, 1, 1);
    
    QWidget* container = new QWidget(parent);
    container->setLayout(layout);
    return container;
  }
  
  QLineEdit* black_value_edit_;
  QLineEdit* white_value_edit_;
  
  ImageDisplayQtWindow* window_;
};



ImageDisplayQtWindow::ImageDisplayQtWindow(
    ImageDisplay* display,
    QWidget* parent,
    Qt::WindowFlags flags)
    : QMainWindow(parent, flags),
      last_zoom_factor_(1),
      have_last_cursor_pos_(false),
      last_pixel_value_valid_(false),
      black_value_(0),
      white_value_(255),
      display_(display) {
  InitQtResources();
  
  // Create main layout with image widget.
  QVBoxLayout* layout = new QVBoxLayout();
  layout->setContentsMargins(0, 0, 0, 0);
  image_widget_ = new ImageDisplayQtWidget(this);
  connect(image_widget_, &ImageDisplayQtWidget::CursorPositionChanged, this, &ImageDisplayQtWindow::CursorPositionChanged);
  connect(image_widget_, &ImageDisplayQtWidget::ZoomChanged, this, &ImageDisplayQtWindow::ZoomChanged);
  layout->addWidget(image_widget_, 1);
  
  // Embed the layout in a widget used as central widget.
  QWidget* main_widget = new QWidget();
  main_widget->setLayout(layout);
  main_widget->setAutoFillBackground(false);
  setCentralWidget(main_widget);
  
  // Create the toolbar.
  tool_bar_ = new QToolBar("Main toolbar", this);
  tool_bar_->addAction(QIcon(":/save.png"), "&Save image", this, SLOT(SaveImage()));
  tool_bar_->addAction(QIcon(":/copy.png"), "&Copy image to clipboard", this, SLOT(CopyImage()));
  tool_bar_->addAction(QIcon(":/scale_1x.png"), "Set the scaling to one", this, SLOT(SetScaleToOne()));
  fit_contents_act_ = tool_bar_->addAction(QIcon(":/fit_contents.png"), "&Fit image to window size", this, SLOT(FitContentToggled()));
  fit_contents_act_->setCheckable(true);
  zoom_and_resize_act = tool_bar_->addAction(QIcon(":/zoom_and_resize_to_contents.png"), "&Zoom and resize window to image", this, SLOT(ZoomAndResizeToContent()));
  resize_act = tool_bar_->addAction(QIcon(":/resize_to_contents.png"), "&Resize window to image", this, SLOT(ResizeToContentWithoutZoom()));
  
  QToolButton* settings_toolbutton = new QToolButton();
  settings_toolbutton->setIcon(QIcon(":/settings_sliders.png"));
  settings_toolbutton->setPopupMode(QToolButton::InstantPopup);
  settings_toolbutton->addAction(new DisplaySettingsAction(this, this));
  tool_bar_->addWidget(settings_toolbutton);
  
  addToolBar(Qt::TopToolBarArea, tool_bar_);
  
  // Enable the status bar.
  statusBar()->show();
  
  // Use resize(0, 0) here to prevent the window from flashing up with a
  // different size than its final one at first.
  resize(0, 0);
}

void ImageDisplayQtWindow::SetBlackWhiteValues(double black, double white) {
  black_value_ = black;
  white_value_ = white;
}

void ImageDisplayQtWindow::SetCallbacks(const shared_ptr<ImageWindowCallbacks>& callbacks) {
  if (callbacks) {
    callbacks->SetRenderWindow(this);
  }
  
  image_widget_->SetCallbacks(callbacks);
}

void ImageDisplayQtWindow::SetDisplay(ImageDisplay* display) {
  display_ = display;
}

void ImageDisplayQtWindow::AddSubpixelDotPixelCornerConv(float x, float y, u8 r, u8 g, u8 b) {
  image_widget_->AddSubpixelDotPixelCornerConv(x, y, r, g, b);
}

void ImageDisplayQtWindow::AddSubpixelLinePixelCornerConv(float x0, float y0, float x1, float y1, u8 r, u8 g, u8 b) {
  image_widget_->AddSubpixelLinePixelCornerConv(x0, y0, x1, y1, r, g, b);
}

void ImageDisplayQtWindow::AddSubpixelTextPixelCornerConv(float x, float y, u8 r, u8 g, u8 b, const string& text) {
  image_widget_->AddSubpixelTextPixelCornerConv(x, y, r, g, b, text);
}

void ImageDisplayQtWindow::Clear() {
  image_widget_->Clear();
}

void ImageDisplayQtWindow::SetDisplayAsWidget() {
  setWindowFlags(Qt::Widget);
  zoom_and_resize_act->setVisible(false);
  resize_act->setVisible(false);
}

void ImageDisplayQtWindow::CursorPositionChanged(const QPointF& pixel_pos, bool pixel_value_valid, const std::string& pixel_value, QRgb pixel_displayed_value) {
  last_cursor_pos_ = pixel_pos;
  last_pixel_value_valid_ = pixel_value_valid;
  last_pixel_value_ = pixel_value;
  last_pixel_displayed_value_ = pixel_displayed_value;
  have_last_cursor_pos_ = true;
  UpdateStatusBar();
}

void ImageDisplayQtWindow::ZoomChanged(double zoom_factor) {
  last_zoom_factor_ = zoom_factor;
  UpdateStatusBar();
}

void ImageDisplayQtWindow::SaveImage() {
  QString file_path = QFileDialog::getSaveFileName(this, "Save image");
  if (file_path.isEmpty()) {
    return;
  }
  
  if (image_widget_->image().save(file_path, nullptr, 100)) {
    statusBar()->showMessage("Image saved to: " + file_path, 3000);
  } else {
    QMessageBox::warning(this, "Error", "Failed to save image to: " + file_path);
  }
}

void ImageDisplayQtWindow::CopyImage() {
  QClipboard* clipboard = QApplication::clipboard();
  clipboard->setImage(image_widget_->image());
}

void ImageDisplayQtWindow::ResizeToContent(bool adjust_zoom) {
  // Get the size of the screen which this window is on.
  QRect screen_size = QDesktopWidget().availableGeometry(this);
  
  QSize size;
  
  if (adjust_zoom) {
    // Set the window size such that the image is fully displayed with 1x zoom,
    // or adjust the zoom if the window does not fit into the screen this way.
    // If the image is too small, start with a larger zoom factor.
    
    // Determine the required size to fit the image in its original resolution.
    double initial_zoom_factor = 1.0;
    size = QSize(
        initial_zoom_factor * image_widget_->sizeHint().width(),
        initial_zoom_factor * image_widget_->sizeHint().height() +
            statusBar()->height() + tool_bar_->height());
    
    // If the size is smaller than a threshold in any of the two dimensions,
    // start zoomed in.
    constexpr int kMinWidth = 200;
    constexpr int kMinHeight = 100;
    while (size.width() < kMinWidth) {
      initial_zoom_factor *= 2;
      size = QSize(
        initial_zoom_factor * image_widget_->sizeHint().width(),
        initial_zoom_factor * image_widget_->sizeHint().height() +
            statusBar()->height() + tool_bar_->height());
    }
    while (size.height() < kMinHeight) {
      initial_zoom_factor *= 2;
      size = QSize(
        initial_zoom_factor * image_widget_->sizeHint().width(),
        initial_zoom_factor * image_widget_->sizeHint().height() +
            statusBar()->height() + tool_bar_->height());
    }
    
    // If the size is larger than the screen size in any of the two dimensions,
    // zoom out until it fits.
    while (size.width() > screen_size.width() ||
           size.height() > screen_size.height()) {
      initial_zoom_factor *= 0.5;
      size = QSize(
        initial_zoom_factor * image_widget_->sizeHint().width(),
        initial_zoom_factor * image_widget_->sizeHint().height() +
            statusBar()->height() + tool_bar_->height());
    }
    
    image_widget_->SetZoomFactor(initial_zoom_factor);
  } else {
    size = QSize(
        last_zoom_factor_ * image_widget_->sizeHint().width(),
        last_zoom_factor_ * image_widget_->sizeHint().height() +
            statusBar()->height() + tool_bar_->height());
    
    if (size.width() >= screen_size.width()) {
      size.setWidth(screen_size.width());
    }
    
    if (size.height() >= screen_size.height()) {
      size.setHeight(screen_size.height());
    }
  }
  
  // Resize the window to fit the image with the initial zoom factor.
  image_widget_->SetViewOffset(0, 0);
  
  if (!isMaximized()) {
    resize(size);
  }
}

void ImageDisplayQtWindow::SetScaleToOne() {
  image_widget_->ZoomAt(image_widget_->width() / 2, image_widget_->height() / 2, 1);
}

void ImageDisplayQtWindow::FitContent(bool enable) {
  fit_contents_act_->setChecked(enable);
  if (enable) {
    image_widget_->FitContent();
  }
}

void ImageDisplayQtWindow::FitContentToggled() {
  if (fit_contents_act_->isChecked()) {
    FitContent();
  }
}

void ImageDisplayQtWindow::closeEvent(QCloseEvent* event) {
  if (display_) {
    display_->SetWindow(nullptr);
  }
  event->accept();
}

void ImageDisplayQtWindow::UpdateStatusBar() {
  QString zoom_string = "Zoom: " + QString::number(last_zoom_factor_, 'f', 3);
  QString pixel_value_string;
  if (last_pixel_value_valid_) {
    pixel_value_string =
        "Image value: " + QString::fromStdString(last_pixel_value_) + ", " +
        "Displayed value: (" + QString::number(qRed(last_pixel_displayed_value_)) +
        ", " + QString::number(qGreen(last_pixel_displayed_value_)) +
        ", " + QString::number(qBlue(last_pixel_displayed_value_)) + ")";
  }
  QString cursor_pos_string;
  if (have_last_cursor_pos_) {
    cursor_pos_string =
        "Cursor: " + QString::number(last_cursor_pos_.x(), 'f', 1) + ", " +
        QString::number(last_cursor_pos_.y(), 'f', 1);
  }
  
  if (have_last_cursor_pos_ && last_pixel_value_valid_) {
    statusBar()->showMessage(cursor_pos_string + ", " + pixel_value_string + ", " + zoom_string);
  } else if (have_last_cursor_pos_ && !last_pixel_value_valid_) {
    statusBar()->showMessage(cursor_pos_string + ", " + zoom_string);
  } else {
    statusBar()->showMessage(zoom_string);
  }
}

}

#include "image_display_qt_window.moc"
