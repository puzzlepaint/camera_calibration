// Copyright 2011-2013 Paul Furgale and others, 2017, 2019 ETH Zürich, Thomas Schöps
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

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "libvis/libvis.h"

namespace vis {

/// Measures the elapsed time between starting and stopping the timer. Measures
/// real (wall clock) time, as opposed to for example processor time.
/// TODO: Regarding the API, would it be more intuitive to have "start" as a
///       parameter of the constructors instead of "construct_stopped"?
///       --> One would pass true to start the timer, false to not start it.
class Timer {
 public:
  /// Constructs a timer. By default, it is started immediately.
  Timer(bool construct_stopped = false);
  
  /// Constructs a timer assigned to a handle. The handle is used to compile
  /// statistics of timers with the same handle. It must be obtained from the
  /// Timing singleton instance. The statistics can then also be obtained from
  /// that singleton. By default, the timer is started immediately.
  Timer(usize handle, bool construct_stopped = false);
  
  /// Constructs a timer assigned to the handle identified by the given tag.
  /// The handle is used to compile statistics of timers with the same handle.
  /// The statistics can then be obtained from the Timing singleton instance.
  /// By default, the timer is started immediately.
  Timer(const string& tag, bool construct_stopped = false);
  
  /// Constructs a timer assigned to the handle identified by the given tag.
  /// The handle is used to compile statistics of timers with the same handle.
  /// The statistics can then be obtained from the Timing singleton instance.
  /// By default, the timer is started immediately.
  Timer(const char* tag, bool construct_stopped = false);
  
  /// Destructor. Stops the timer if it has not been stopped by manually calling
  /// Stop() yet.
  ~Timer();
  
  /// Starts the timer. Can only be called if the timer is not currently running.
  void Start();
  
  /// Stops the timer. Returns the elapsed time since the start in seconds.
  double Stop(bool add_to_statistics = true);
  
  /// Returns the elapsed time since the start in seconds, but unlike Stop(),
  /// does not stop the timer.
  double GetTimeSinceStart();
  
  /// Returns whether the timer is currently running.
  inline bool IsTiming() const { return timing_; };

 private:
  chrono::steady_clock::time_point start_time_;
  bool timing_;
  usize handle_;
};

class DisabledTimer {
 public:
  inline DisabledTimer(usize /*handle*/, bool /*construct_stopped*/ = false) {}
  inline DisabledTimer(const string& /*tag*/, bool /*construct_stopped*/ = false) {}
  inline ~DisabledTimer() {}
  
  inline void Start() {}
  inline double Stop(bool /*add_to_statistics*/ = true) { return 0; }
  inline bool IsTiming() { return false; }
};

#ifdef LIBVIS_ENABLE_TIMING
typedef Timer ConditionalTimer;
#else
typedef DisabledTimer ConditionalTimer;
#endif

enum SortType {kSortByTotal, kSortByMean, kSortByStd, kSortByMin, kSortByMax, kSortByNumSamples};

struct TimerMapValue;

class Timing {
 public:
  static void addTime(usize handle, double seconds);
  
  static usize getHandle(const string& tag);
  static string getTag(usize handle);
  static double getTotalSeconds(usize handle);
  static double getTotalSeconds(const string& tag);
  static double getMeanSeconds(usize handle);
  static double getMeanSeconds(const string& tag);
  static usize getNumSamples(usize handle);
  static usize getNumSamples(const string& tag);
  static double getVarianceSeconds(usize handle);
  static double getVarianceSeconds(const string& tag);
  static double getMinSeconds(usize handle);
  static double getMinSeconds(const string& tag);
  static double getMaxSeconds(usize handle);
  static double getMaxSeconds(const string& tag);
  static double getHz(usize handle);
  static double getHz(const string& tag);
  static void print(ostream& out);
  static void print(ostream& out, const SortType sort);
  static void reset(usize handle);
  static void reset(const string& tag);
  static string print();
  static string print(const SortType sort);
  static string secondsToTimeString(double seconds, bool long_format = false);
  
 private:
  template <typename TMap, typename Accessor>
  static void print(const TMap& map, const Accessor& accessor, ostream& out);
  
  static Timing& instance();
  
  // Singleton design pattern
  Timing();
  ~Timing();
  
  typedef unordered_map<string, usize> map_t;
  typedef vector<TimerMapValue> list_t;
  
  // Static members
  list_t m_timers;
  map_t m_tagMap;
#ifdef SM_USE_HIGH_PERF_TIMER
  double m_clockPeriod;
#endif
  usize m_maxTagLength;
    
  static mutex m_mutex;
};

}
