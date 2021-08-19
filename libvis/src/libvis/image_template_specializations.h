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
bool Image<float>::Write(const string& image_file_name) const {
  std::ofstream out(image_file_name + ".data");
  if (!out.good()) {
    return false;
  }
  for(int xx = 0; xx < width(); ++xx) {
    for (int yy = 0; yy < height(); ++yy) {
      out << xx << "\t" << yy << "\t" << operator()(xx,yy) << std::endl;
      if (!out.good()) {
        return false;
      }
    }
    out << std::endl;
  }
  std::ofstream gnuplot(image_file_name + ".gpl");
  gnuplot << "set term svg enhanced background rgb 'white';\n"
          << "set output '" << image_file_name << ".svg';\n"
          << "set view equal xy;\n"
          << "set xrange [0:" << width() << "];\n"
          << "set yrange [0:" << height() << "];\n"
          << "set xtics out;\n"
          << "set ytics out;\n"
          << "set title 'Reprojection error magnitude [px]';\n"
          << "plot '" << image_file_name << ".data' with image notitle;\n";

  return true;
}

template<>
bool Image<Vec2f>::Write(const string& image_file_name) const {
  std::ofstream out(image_file_name, std::ofstream::binary);
  if (!out.good())
      return false;

  int const width = this->height();
  int const height = this->width();

  out.write("PIEH", 4);
  out.write(reinterpret_cast<const char*>(&height), sizeof(height));
  out.write(reinterpret_cast<const char*>(&width), sizeof(width));
  if (!out.good())
      return false;

  for (int row = 0; row < width; row++ ) {
    for (int col = 0; col < height; ++col) {
      float const x = operator()(col, row).x();
      float const y = operator()(col, row).y();
      out.write(reinterpret_cast<const char*>(&x), sizeof x);
      out.write(reinterpret_cast<const char*>(&y), sizeof y);
      if (!out.good()) {
        return false;
      }
    }
  }
  out.close();
  return true;
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
