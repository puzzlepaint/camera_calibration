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


#include "libvis/qt_thread.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

#include <QApplication>
#include <QThread>
#include <QTimer>

#include "libvis/logging.h"

namespace vis {

void RunInQtThread(const function<void()>& f) {
  if (!qApp) {
    LOG(ERROR) << "RunInQtThread(): No qApp exists. Not running the function.";
    return;
  }
  if (QThread::currentThread() == qApp->thread()) {
    f();
    return;
  }  
  
  QTimer* timer = new QTimer();
  timer->moveToThread(qApp->thread());
  timer->setSingleShot(true);
  QObject::connect(timer, &QTimer::timeout, [=]() {
    f();
    timer->deleteLater();
  });
  QMetaObject::invokeMethod(timer, "start", Qt::QueuedConnection, Q_ARG(int, 0));
}

void RunInQtThreadBlocking(const function<void()>& f) {
  if (!qApp) {
    LOG(ERROR) << "RunInQtThreadBlocking(): No qApp exists. Not running the function.";
    return;
  }
  if (QThread::currentThread() == qApp->thread()) {
    f();
    return;
  }
  
  mutex done_mutex;
  condition_variable done_condition;
  atomic<bool> done;
  done = false;
  
  QTimer* timer = new QTimer();
  timer->moveToThread(qApp->thread());
  timer->setSingleShot(true);
  QObject::connect(timer, &QTimer::timeout, [&]() {
    f();
    timer->deleteLater();
    
    lock_guard<mutex> lock(done_mutex);
    done = true;
    done_condition.notify_all();
  });
  QMetaObject::invokeMethod(timer, "start", Qt::QueuedConnection, Q_ARG(int, 0));
  
  unique_lock<mutex> lock(done_mutex);
  while (!done) {
    done_condition.wait(lock);
  }
}

int WrapQtEventLoopAround(function<int (int, char**)> func, int argc, char** argv) {
  QApplication qapp(argc, argv);
  qapp.setQuitOnLastWindowClosed(false);
  
  // Start the actual application in its own thread
  int return_value = 1;
  thread app_thread([&]{
    return_value = func(argc, argv);
    RunInQtThreadBlocking([&]() {
      qapp.closeAllWindows();
    });
    qapp.quit();
  });
  
  // Run the Qt event loop
  qapp.exec();
  
  app_thread.join();
  return return_value;
}

}
