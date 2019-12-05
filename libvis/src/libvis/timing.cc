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


#include "libvis/timing.h"

#include <algorithm>
#include <limits>
#include <map>
#include <math.h>

#include "libvis/logging.h"

namespace vis {

Timer::Timer(bool construct_stopped)
    : timing_(false),
      handle_(numeric_limits<usize>::max()) {
  if (!construct_stopped) {
    Start();
  }
}

Timer::Timer(usize handle, bool construct_stopped)
    : timing_(false),
      handle_(handle) {
  if (!construct_stopped) {
    Start();
  }
}

Timer::Timer(const string& tag, bool construct_stopped)
    : timing_(false),
      handle_(Timing::getHandle(tag)) {
  if (!construct_stopped) {
    Start();
  }
}

Timer::Timer(const char* tag, bool construct_stopped)
    : timing_(false),
      handle_(Timing::getHandle(tag)) {
  if (!construct_stopped) {
    Start();
  }
}

Timer::~Timer() {
  if (IsTiming()) {
    Stop();
  }
}

void Timer::Start() {
  CHECK(!IsTiming());
  
  timing_ = true;
  start_time_ = chrono::steady_clock::now();
}

double Timer::Stop(bool add_to_statistics) {
  double seconds = GetTimeSinceStart();
  if (add_to_statistics && handle_ != numeric_limits<usize>::max()) {
    Timing::addTime(handle_, seconds);
  }
  timing_ = false;
  return seconds;
}

double Timer::GetTimeSinceStart() {
  CHECK(timing_) << "GetTimeSinceStart() called on a stopped timer";
  chrono::steady_clock::time_point now = chrono::steady_clock::now();
  double seconds = 1e-9 * chrono::duration<double, nano>(now - start_time_).count();
  return seconds;
}

// Algorithm from:
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
struct TimerMapValue {
  TimerMapValue()
      : count(0),
        min(numeric_limits<double>::infinity()),
        max(-numeric_limits<double>::infinity()),
        M2(0),
        mean(0) {}
  
  void AddValue(double x) {
    count += 1;
    double delta = x - mean;
    mean += delta / count;
    double delta2 = x - mean;
    M2 += delta * delta2;
    
    if (x < min) {
      min = x;
    }
    if (x > max) {
      max = x;
    }
  }
  
  double GetVariance() const {
    if (count < 2) {
      return 0;
    } else {
      return M2 / (count - 1);
    }
  }
  
  double GetTotal() const {
    return count * mean;
  }
  
  usize count;
  double min;
  double max;
  double M2;
  double mean;
};


mutex Timing::m_mutex;

Timing& Timing::instance() {
  static Timing t;
  return t;
}

Timing::Timing() :
    m_maxTagLength(0) {}

Timing::~Timing() {}

usize Timing::getHandle(string const& tag){
  // Search for an existing tag.
  unique_lock<mutex> lock(m_mutex);
  map_t::iterator i = instance().m_tagMap.find(tag);
  if (i == instance().m_tagMap.end()) {
    // If it is not there, create a tag.
    usize handle = instance().m_timers.size();
    instance().m_tagMap[tag] = handle;
    instance().m_timers.push_back(TimerMapValue());
    // Track the maximum tag length to help printing a table of timing values later.
    instance().m_maxTagLength = std::max(instance().m_maxTagLength, tag.size());
    return handle;
  } else {
    return i->second;
  }
}

string Timing::getTag(usize handle){
  string tag;
  bool found = false;
  
  // Perform a linear search for the tag
  map_t::iterator i = instance().m_tagMap.begin();
  for ( ; i != instance().m_tagMap.end(); i++) {
    if (i->second == handle){
      found = true;
      tag = i->first;
      break;
    }
  }
  
  CHECK(found) << "Unable to find the tag associated with handle " << handle;
  return tag;
}

void Timing::addTime(usize handle, double seconds) {
  unique_lock<mutex> lock(m_mutex);
  instance().m_timers[handle].AddValue(seconds);
}

double Timing::getTotalSeconds(usize handle) {
  CHECK_LT(handle, instance().m_timers.size()) << "Handle is out of range: " << handle << ", number of timers: " << instance().m_timers.size();
  return instance().m_timers[handle].GetTotal();
}

double Timing::getTotalSeconds(string const& tag) {
  return getTotalSeconds(getHandle(tag));
}

double Timing::getMeanSeconds(usize handle) {
  CHECK_LT(handle, instance().m_timers.size()) << "Handle is out of range: " << handle << ", number of timers: " << instance().m_timers.size();
  return instance().m_timers[handle].mean;
}

double Timing::getMeanSeconds(string const& tag) {
  return getMeanSeconds(getHandle(tag));
}

usize Timing::getNumSamples(usize handle) {
  CHECK_LT(handle, instance().m_timers.size()) << "Handle is out of range: " << handle << ", number of timers: " << instance().m_timers.size();
  return instance().m_timers[handle].count;
}

usize Timing::getNumSamples(string const& tag) {
  return getNumSamples(getHandle(tag));
}

double Timing::getVarianceSeconds(usize handle) {
  CHECK_LT(handle, instance().m_timers.size()) << "Handle is out of range: " << handle << ", number of timers: " << instance().m_timers.size();
  return instance().m_timers[handle].GetVariance();
}

double Timing::getVarianceSeconds(string const& tag) {
  return getVarianceSeconds(getHandle(tag));
}

double Timing::getMinSeconds(usize handle) {
  CHECK_LT(handle, instance().m_timers.size()) << "Handle is out of range: " << handle << ", number of timers: " << instance().m_timers.size();
  return instance().m_timers[handle].min;
}

double Timing::getMinSeconds(string const& tag) {
  return getMinSeconds(getHandle(tag));
}

double Timing::getMaxSeconds(usize handle) {
  CHECK_LT(handle, instance().m_timers.size()) << "Handle is out of range: " << handle << ", number of timers: " << instance().m_timers.size();
  return instance().m_timers[handle].max;
}

double Timing::getMaxSeconds(string const& tag) {
  return getMaxSeconds(getHandle(tag));
}

double Timing::getHz(usize handle) {
  CHECK_LT(handle, instance().m_timers.size()) << "Handle is out of range: " << handle << ", number of timers: " << instance().m_timers.size();
  return 1.0 / instance().m_timers[handle].mean;
}

double Timing::getHz(string const& tag) {
  return getHz(getHandle(tag));
}

void Timing::reset(usize handle) {
  unique_lock<mutex> lock(m_mutex);
  CHECK_LT(handle, instance().m_timers.size()) << "Handle is out of range: " << handle << ", number of timers: " << instance().m_timers.size();
  instance().m_timers[handle] = TimerMapValue();
}

void Timing::reset(string const& tag) {
  return reset(getHandle(tag));
}

string Timing::secondsToTimeString(double seconds, bool long_format) {
  
//   double secs = fmod(seconds,60);
//   int minutes = (long)(seconds/60);
//   int hours = (long)(seconds/3600);
//   minutes = minutes - (hours*60);
//   
//   char buffer[256];
// #ifdef WIN32
//   sprintf_s(buffer,256,"%02d:%02d:%09.6f",hours,minutes,secs);
// #else
//   sprintf(buffer,"%02d:%02d:%09.6f",hours,minutes,secs);
// #endif
  
  char buffer[256];
#ifdef WIN32
  sprintf_s(buffer, 256, long_format ? "%011.4f" : "%09.6f", seconds);
#else
  sprintf(buffer, long_format ? "%011.4f" : "%09.6f", seconds);
#endif
  return buffer;
}

template <typename TMap, typename Accessor>
void Timing::print(const TMap& tagMap, const Accessor& accessor, ostream& out) {
  out << "Timing\n";
  out << "------\n";
  for (typename TMap::const_iterator t = tagMap.begin(); t != tagMap.end(); ++ t) {
    usize i = accessor.getIndex(t);
    if (getNumSamples(i) == 0) {
      continue;
    }
    
    out.width((streamsize)instance().m_maxTagLength);
    out.setf(ios::left,ios::adjustfield);
    out << accessor.getTag(t) << "\t";
    
    out.width(8);
    out.setf(ios::right,ios::adjustfield);
    out << getNumSamples(i) << "\t";
    if (getNumSamples(i) > 0) {
      out << secondsToTimeString(getTotalSeconds(i), true) << "\t";
      double meansec = getMeanSeconds(i);
      double stddev = sqrt(getVarianceSeconds(i));
      out << "(" << secondsToTimeString(meansec) << " +- ";
      out << secondsToTimeString(stddev) << ")\t";

      double minsec = getMinSeconds(i);
      double maxsec = getMaxSeconds(i);

      // The min or max are out of bounds.
      out << "[" << secondsToTimeString(minsec) << "," << secondsToTimeString(maxsec) << "]";

    }
    out << endl;
  }
}

void Timing::print(ostream& out) {
  struct Accessor {
    usize getIndex(map_t::const_iterator t) const {
      return t->second;
    }
    const string&  getTag(map_t::const_iterator t) const {
      return t->first;
    }
  };

  print(instance().m_tagMap, Accessor(), out);
}

void Timing::print(ostream& out, const SortType sort) {
  map_t& tagMap = instance().m_tagMap;

  typedef multimap<double, string, greater<double> > SortMap_t;
  SortMap_t sorted;
  for(map_t::const_iterator t = tagMap.begin(); t != tagMap.end(); t++) {
    usize i = t->second;
    double sv;
    if(getNumSamples(i) > 0)
      switch (sort) {
        case kSortByTotal:
          sv = getTotalSeconds(i);
          break;
        case kSortByMean:
          sv = getMeanSeconds(i);
          break;
        case kSortByStd:
          sv = sqrt(getVarianceSeconds(i));
          break;
        case kSortByMax:
          sv = getMaxSeconds(i);
          break;
        case kSortByMin:
          sv = getMinSeconds(i);
          break;
        case kSortByNumSamples:
          sv = getNumSamples(i);
          break;
      }
    else
      sv = numeric_limits<double>::max();
    sorted.insert(SortMap_t::value_type(sv, t->first));
  }

  struct Accessor {
    map_t& tagMap;
    Accessor(map_t& tagMap) : tagMap(tagMap) {}
    
    usize getIndex(SortMap_t::const_iterator t) const {
      return tagMap[t->second];
    }
    const string& getTag(SortMap_t::const_iterator t) const {
      return t->second;
    }
  };
  
  print(sorted, Accessor(tagMap), out);
}

string Timing::print()
{
  stringstream ss;
  print(ss);
  return ss.str();
}

string Timing::print(const SortType sort)
{
  stringstream ss;
  print(ss, sort);
  return ss.str();
}
}
