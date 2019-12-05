// This file is included in image.h or image.cc depending on the compiler.

template<>
bool Image<u8>::Write(const string& image_file_name) const {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Write(image_file_name, *this);
}

template<>
bool Image<u16>::Write(const string& image_file_name) const {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Write(image_file_name, *this);
}

template<>
bool Image<Vec3u8>::Write(const string& image_file_name) const {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Write(image_file_name, *this);
}

template<>
bool Image<Vec4u8>::Write(const string& image_file_name) const {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Write(image_file_name, *this);
}

template<>
bool Image<u8>::Read(const string& image_file_name) {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Read(image_file_name, this);
}

template<>
bool Image<u16>::Read(const string& image_file_name) {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Read(image_file_name, this);
}

template<>
bool Image<Vec3u8>::Read(const string& image_file_name) {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Read(image_file_name, this);
}

template<>
bool Image<Vec4u8>::Read(const string& image_file_name) {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Read(image_file_name, this);
}

#ifdef LIBVIS_HAVE_QT
template<>
QImage Image<u8>::WrapInQImage() const {
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
  return WrapInQImage(QImage::Format_Grayscale8);
#else
  // QImage::Format_Grayscale8 not supported. Create a grayscale color table
  // mapping index i to color (i, i, i) to simulate it.
  // NOTE: It would be preferable to make the color table static.
  QVector<QRgb> colors(256);
  for (int i = 0; i < 256; ++i) {
    colors[i] = qRgba(i, i, i, 255);
  }
  QImage image(reinterpret_cast<const u8*>(data()), width(), height(),
               stride(), QImage::Format_Indexed8);
  image.setColorTable(colors);
  return image;
#endif
}

template<>
QImage Image<Vec3u8>::WrapInQImage() const {
  return WrapInQImage(QImage::Format_RGB888);
}

template<>
QImage Image<Vec4u8>::WrapInQImage() const {
  return WrapInQImage(QImage::Format_RGBA8888);
}
#endif
