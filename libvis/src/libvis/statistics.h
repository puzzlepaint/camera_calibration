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

#include <cmath>

#include "libvis/libvis.h"

namespace vis {

/// Implements an algorithm to compute the mean and variance of a set of data
/// points in a single pass over the data. At least two data points are required.
/// 
/// The algorithm is taken from Wikipedia:
///   https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
/// 
/// According to that article, the algorithm comes from:
///   B. P. Welford (1962). "Note on a method for calculating corrected sums of
///   squares and products". Technometrics 4(3):419–420.
template <typename T>
class SinglePassMeanAndVariance {
 public:
  inline SinglePassMeanAndVariance()
      : count_(0),
        mean_(0),
        M2_(0) {}
  
  inline void AddData(T x) {
    ++ count_;
    
    T delta = x - mean_;
    mean_ += delta / count_;
    
    T delta2 = x - mean_;
    M2_ += delta * delta2;
  }
  
  /// Returns the computed mean given the data points added so far. At
  /// least one data point is required to get a valid result.
  inline T mean() const {
    return mean_;
  }
  
  /// Returns the computed variance given the data points added so far.
  /// At least two data points are required to get a valid result.
  /// NOTE: In contrast to mean(), this function performs a (small) computation,
  /// so it should not be called repeatedly in a loop if the result can be
  /// cached.
  inline T ComputeVariance() const {
    return M2_ / (count_ - 1);
  }
  
  /// Returns the number of data points that have been added with AddData().
  inline u32 count() const {
    return count_;
  }
  
 private:
  u32 count_;
  T mean_;
  T M2_;
};


/// Implements computation of the arithmetic mean.
template <typename T>
class Mean {
 public:
  inline Mean()
      : count_(0),
        sum_(0) {}
  
  inline void AddData(T x) {
    ++ count_;
    
    sum_ += x;
  }
  
  /// Returns the arithmetic mean.
  inline T ComputeArithmeticMean() const {
    return sum_ / count_;
  }
  
  /// Returns the number of data points that have been added with AddData().
  inline u32 count() const {
    return count_;
  }
  
 private:
  u32 count_;
  T sum_;
};


/// Implements computation of the geometric mean.
template <typename T>
class GeometricMean {
 public:
  inline GeometricMean()
      : count_(0),
        sum_(0) {}
  
  inline void AddData(T x) {
    ++ count_;
    
    sum_ += log(x);
  }
  
  /// Returns the geometric mean.
  inline T ComputeGeometricMean() const {
    return exp(sum_ / count_);
  }
  
  /// Returns the number of data points that have been added with AddData().
  inline u32 count() const {
    return count_;
  }
  
 private:
  u32 count_;
  T sum_;
};

}
