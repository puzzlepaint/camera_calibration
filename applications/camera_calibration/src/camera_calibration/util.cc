// Copyright 2019 ETH Zurich, Thomas Schöps
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


#include "camera_calibration/util.h"

#include <cstdio>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include <libvis/lm_optimizer.h>
#include <libvis/sophus.h>

#include "camera_calibration/models/central_generic.h"
#include "camera_calibration/models/central_opencv.h"
#include "camera_calibration/models/central_radial.h"
#include "camera_calibration/models/central_thin_prism_fisheye.h"
#include "camera_calibration/models/noncentral_generic.h"

namespace vis {

// Implementation from: https://stackoverflow.com/questions/421860
char GetKeyInput() {
  char buf = 0;
  struct termios old = {0};
  if (tcgetattr(0, &old) < 0) {
    perror("tcsetattr()");
  }
  old.c_lflag &= ~ICANON;
  old.c_lflag &= ~ECHO;
  old.c_cc[VMIN] = 1;
  old.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSANOW, &old) < 0) {
    perror("tcsetattr ICANON");
  }
  if (read(0, &buf, 1) < 0) {
    perror("read()");
  }
  old.c_lflag |= ICANON;
  old.c_lflag |= ECHO;
  if (tcsetattr(0, TCSADRAIN, &old) < 0) {
    perror("tcsetattr ~ICANON");
  }
  return (buf);
}

int PollKeyInput() {
  int character;
  struct termios orig_term_attr;
  struct termios new_term_attr;
  
  /* set the terminal to raw mode */
  tcgetattr(fileno(stdin), &orig_term_attr);
  memcpy(&new_term_attr, &orig_term_attr, sizeof(struct termios));
  new_term_attr.c_lflag &= ~(ECHO|ICANON);
  new_term_attr.c_cc[VTIME] = 0;
  new_term_attr.c_cc[VMIN] = 0;
  tcsetattr(fileno(stdin), TCSANOW, &new_term_attr);
  
  /* read a character from the stdin stream without blocking */
  /*   returns EOF (-1) if no character is available */
  character = fgetc(stdin);
  
  if (character != EOF) {
    std::getchar();  // remove the character from the buffer
  }
  
  /* restore the original terminal attributes */
  tcsetattr(fileno(stdin), TCSANOW, &orig_term_attr);
  
  return character;
}

struct SO3dState {
  /// Initializes the transformation to identity.
  inline SO3dState() {}
  
  inline SO3dState(const SO3dState& other)
      : dest_R_src(other.dest_R_src) {}
  
  inline int degrees_of_freedom() const {
    return SO3d::DoF;
  }
  
  static constexpr bool is_reversible() { return true; }
  
  template <typename Derived>
  inline void operator-=(const MatrixBase<Derived>& delta) {
    // Using minus delta here since we are subtracting. Using
    // left-multiplication of the delta must be consistent with the way the
    // Jacobian is computed.
    dest_R_src = SO3d::exp(-delta) * dest_R_src;
  }
  
  SO3d dest_R_src;
};

struct MatchedPointsDistanceCostFunction {
  const vector<Vec3d>* src_points;
  const vector<Vec3d>* dest_points;
  
  template<bool compute_jacobians, class Accumulator, class State>
  inline void Compute(
      const State& state,
      Accumulator* accumulator) const {
    for (usize i = 0, size = src_points->size(); i < size; ++ i) {
      const Vec3d& src = src_points->at(i);
      const Vec3d& dest_data = dest_points->at(i);
      
      /// Transformed source point.
      const Vec3d dest_state = state.dest_R_src * src;
      
      /// Residuals: (T * src)[c] - dest[c] for each of the 3 vector components
      /// c. Together, this is the squared norm of the vector difference.
      const Vec3d residuals = dest_state - dest_data;
      if (compute_jacobians) {
        // Jacobian of: exp(hat(delta)) * T * src - dest , wrt. delta.
        // The raw Jacobian is:
        // 0, dest_state(2), -dest_state(1)
        // -dest_state(2), 0, dest_state(0)
        // dest_state(1), -dest_state(0), 0
        // The AddJacobian() calls avoid including most of the zeros.
        accumulator->AddResidualWithJacobian(residuals(0), 1, Matrix<double, 1, 2>(dest_state(2), -dest_state(1)));
        accumulator->AddResidualWithJacobian(residuals(1), 0, Matrix<double, 1, 3>(-dest_state(2), 0, dest_state(0)));
        accumulator->AddResidualWithJacobian(residuals(2), 0, Matrix<double, 1, 2>(dest_state(1), -dest_state(0)));
      } else {
        accumulator->AddResidual(residuals(0));
        accumulator->AddResidual(residuals(1));
        accumulator->AddResidual(residuals(2));
      }
    }
  }
};

Mat3d DeterminePointCloudRotation(const vector<Vec3d>& a, const vector<Vec3d>& b) {
  LMOptimizer<double> optimizer;
  
  SO3dState state;
  state.dest_R_src = SO3d();
  
  // Verify that the analytical Jacobian at this state equals the numerical
  // Jacobian (with default step size and precision threshold for all state
  // components).
  MatchedPointsDistanceCostFunction cost_function;
  cost_function.src_points = &b;
  cost_function.dest_points = &a;
  
  // Run the optimization.
  optimizer.Optimize(
      &state,
      cost_function,
      /*max_iteration_count*/ 100);
  
  return state.dest_R_src.matrix();
}

bool CreateObservationDirectionsImage(const CameraModel* model, Image<Vec3d>* dense_model) {
  if (CameraModel::IsCentral(model->type())) {
    constexpr int kBorderX = 0;
    constexpr int kBorderY = 0;
    dense_model->SetSize(model->width() - 2 * kBorderX,
                         model->height() - 2 * kBorderY);
    for (u32 y = 0; y < dense_model->height(); ++ y) {
      for (u32 x = 0; x < dense_model->width(); ++ x) {
        if (!model->Unproject(kBorderX + x + 0.5f, kBorderY + y + 0.5f, &dense_model->at(x, y))) {
          (*dense_model)(x, y) = Vec3d::Constant(numeric_limits<double>::quiet_NaN());
        }
      }
    }
    return true;
  }
  
  const NoncentralGenericModel* ngbsp_model = dynamic_cast<const NoncentralGenericModel*>(model);
  if (ngbsp_model) {
    LOG(WARNING) << "CreateObservationDirectionsImage() called on non-central camera model, ignoring line origins";
    
    constexpr int kBorderX = 0;
    constexpr int kBorderY = 0;
    dense_model->SetSize(model->width() - 2 * kBorderX,
                         model->height() - 2 * kBorderY);
    for (u32 y = 0; y < dense_model->height(); ++ y) {
      for (u32 x = 0; x < dense_model->width(); ++ x) {
        Line3d result;
        
        if (ngbsp_model->Unproject(kBorderX + x + 0.5f, kBorderY + y + 0.5f, &result)) {
          (*dense_model)(x, y) = result.direction();
        } else {
          (*dense_model)(x, y) = Vec3d::Constant(numeric_limits<double>::quiet_NaN());
        }
      }
    }
    return true;
  }
  
  return false;
}

}
