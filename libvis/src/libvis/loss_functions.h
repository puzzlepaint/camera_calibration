// Copyright 2019 ETH Zürich, Thomas Schöps
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

namespace vis {

// This file contains loss function implementations for use with the LMOptimizer
// class. They follow this scheme:
// 
// class Loss {
//  public:
//   /// Given a residual, computes its contribution to the optimization cost.
//   template <typename Scalar>
//   inline Scalar ComputeCost(Scalar residual) const {
//     ...
//   }
//   
//   template <typename Scalar>
//   inline Scalar ComputeCostFromSquaredResidual(Scalar squared_residual) const {
//     ...
//   }
//   
//   /// Given a residual, computes its weight within iterative re-weighted least
//   /// squares (IRLS) optimization. This is equal to:
//   /// (1 / residual) * (d ComputeCost(residual)) / (d residual) .
//   template <typename Scalar>
//   inline Scalar ComputeWeight(Scalar residual) const {
//     ...
//   }
//   
//   template <typename Scalar>
//   inline Scalar ComputeWeightFromSquaredResidual(Scalar squared_residual) const {
//     ...
//   }
//   
// };

/// Standard, non-robust quadratic loss.
class QuadraticLoss {
 public:
  template <typename Scalar>
  inline Scalar ComputeCost(Scalar residual) const {
    return static_cast<Scalar>(0.5) * residual * residual;
  }
  
  template <typename Scalar>
  inline Scalar ComputeCostFromSquaredResidual(Scalar squared_residual) const {
    return static_cast<Scalar>(0.5) * squared_residual;
  }
  
  template <typename Scalar>
  inline Scalar ComputeWeight(Scalar /*residual*/) const {
    return 1;
  }
  
  template <typename Scalar>
  inline Scalar ComputeWeightFromSquaredResidual(Scalar /*squared_residual*/) const {
    return 1;
  }
};

/// Huber's robust loss function, which uses L2 loss close to zero and L1 loss
/// otherwise.
template <typename T>
class HuberLoss {
 public:
  inline HuberLoss(T huber_parameter)
      : m_huber_parameter(huber_parameter) {}
  
  template <typename Scalar>
  inline Scalar ComputeCost(Scalar residual) const {
    const Scalar abs_residual = fabs(residual);
    if (abs_residual < m_huber_parameter) {
      return static_cast<Scalar>(0.5) * residual * residual;
    } else {
      return m_huber_parameter * (abs_residual - static_cast<Scalar>(0.5) * m_huber_parameter);
    }
  }
  
  template <typename Scalar>
  inline Scalar ComputeCostFromSquaredResidual(Scalar squared_residual) const {
    if (squared_residual < m_huber_parameter * m_huber_parameter) {
      return static_cast<Scalar>(0.5) * squared_residual;
    } else {
      return m_huber_parameter * (sqrt(squared_residual) - static_cast<Scalar>(0.5) * m_huber_parameter);
    }
  }
  
  template <typename Scalar>
  inline Scalar ComputeWeight(Scalar residual) const {
    const Scalar abs_residual = fabs(residual);
    return (abs_residual < m_huber_parameter) ? 1 : (m_huber_parameter / abs_residual);
  }
  
  template <typename Scalar>
  inline Scalar ComputeWeightFromSquaredResidual(Scalar squared_residual) const {
    return (squared_residual < m_huber_parameter * m_huber_parameter) ? 1 : (m_huber_parameter / sqrt(squared_residual));
  }
  
 private:
  T m_huber_parameter;
};

/// Tukey's biweight function, which lets the influence of outliers be zero.
template <typename T>
class TukeyBiweightLoss {
 public:
  inline TukeyBiweightLoss(T tukey_parameter)
      : m_tukey_parameter(tukey_parameter),
        m_tukey_parameter_sq(tukey_parameter * tukey_parameter) {}
  
  template <typename Scalar>
  inline Scalar ComputeCost(Scalar residual) const {
    if (fabs(residual) < m_tukey_parameter) {
      const float quot = residual / m_tukey_parameter;
      const float term = 1.f - quot * quot;
      return (1 / 6.f) * m_tukey_parameter_sq * (1 - term * term * term);
    } else {
      return (1 / 6.f) * m_tukey_parameter_sq;
    }
  }
  
  template <typename Scalar>
  inline Scalar ComputeCostFromSquaredResidual(Scalar squared_residual) const {
    if (squared_residual < m_tukey_parameter_sq) {
      const float term = 1.f - squared_residual / m_tukey_parameter_sq;
      return (1 / 6.f) * m_tukey_parameter_sq * (1 - term * term * term);
    } else {
      return (1 / 6.f) * m_tukey_parameter_sq;
    }
  }
  
  template <typename Scalar>
  inline Scalar ComputeWeight(Scalar residual) const {
    if (fabs(residual) < m_tukey_parameter) {
      const float quot = residual / m_tukey_parameter;
      const float term = 1.f - quot * quot;
      return term * term;
    } else {
      return 0.f;
    }
  }
  
  template <typename Scalar>
  inline Scalar ComputeWeightFromSquaredResidual(Scalar squared_residual) const {
    if (squared_residual < m_tukey_parameter_sq) {
      const float term = 1.f - squared_residual / m_tukey_parameter_sq;
      return term * term;
    } else {
      return 0.f;
    }
  }
  
 private:
  T m_tukey_parameter;
  T m_tukey_parameter_sq;
};

/// Cauchy robust loss function (TODO: describe what it does and when one might
/// want to use it).
template <typename T>
class CauchyLoss {
 public:
  inline CauchyLoss(T cauchy_parameter)
      : m_cauchy_parameter(cauchy_parameter),
        m_cauchy_parameter_sq(cauchy_parameter * cauchy_parameter) {}
  
  template <typename Scalar>
  inline Scalar ComputeCost(Scalar residual) const {
    float div = residual / m_cauchy_parameter;
    return 0.5f * m_cauchy_parameter_sq * log(1 + div * div);
  }
  
  template <typename Scalar>
  inline Scalar ComputeCostFromSquaredResidual(Scalar squared_residual) const {
    return 0.5f * m_cauchy_parameter_sq * log(1 + squared_residual / m_cauchy_parameter_sq);
  }
  
  template <typename Scalar>
  inline Scalar ComputeWeight(Scalar residual) const {
    float div = residual / m_cauchy_parameter;
    return 1.f / (1 + div * div);
  }
  
  template <typename Scalar>
  inline Scalar ComputeWeightFromSquaredResidual(Scalar squared_residual) const {
    return 1.f / (1 + squared_residual / m_cauchy_parameter_sq);
  }
  
 private:
  T m_cauchy_parameter;
  T m_cauchy_parameter_sq;
};

}
