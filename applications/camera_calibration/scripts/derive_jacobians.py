import math
import sys
import time

from sympy import *
from sympy.solvers.solveset import nonlinsolve

from optimizer_builder import *


# ### Math functions ###

# Simple model for the fractional-part function used for bilinear interpolation
# which leaves the function un-evaluated. Ignores the discontinuities when
# computing the derivative. They do not matter.
class frac(Function):
  # Returns the first derivative of the function.
  # A simple model for the function within the range between two discontinuities is:
  # f(x) = x - c, with a constant c. So f'(x) = 1.
  def fdiff(self, argindex=1):
    if argindex == 1:
      return S.One
    else:
      raise ArgumentIndexError(self, argindex)

def UnitQuaternionRotatePoint(q, pt):
  t2 =  q[0] * q[1]
  t3 =  q[0] * q[2]
  t4 =  q[0] * q[3]
  t5 = -q[1] * q[1]
  t6 =  q[1] * q[2]
  t7 =  q[1] * q[3]
  t8 = -q[2] * q[2]
  t9 =  q[2] * q[3]
  t1 = -q[3] * q[3]
  return Matrix([[2 * ((t8 + t1) * pt[0] + (t6 - t4) * pt[1] + (t3 + t7) * pt[2]) + pt[0]],
                 [2 * ((t4 + t6) * pt[0] + (t5 + t1) * pt[1] + (t9 - t2) * pt[2]) + pt[1]],
                 [2 * ((t7 - t3) * pt[0] + (t2 + t9) * pt[1] + (t5 + t8) * pt[2]) + pt[2]]])


# Transformation is a 7-vector [quaternion, translation].
def TransformPoint(transformation, point):
  point_out = UnitQuaternionRotatePoint(transformation, point)
  point_out[0] += transformation[4];
  point_out[1] += transformation[5];
  point_out[2] += transformation[6];
  
  return point_out


# Both transformations are 7-vectors [quaternion, translation].
def RigTransformPoint(camera_tr_rig, rig_tr_global, global_point):
  point_rig = UnitQuaternionRotatePoint(rig_tr_global, global_point)
  point_rig[0] += rig_tr_global[4];
  point_rig[1] += rig_tr_global[5];
  point_rig[2] += rig_tr_global[6];
  
  point_out = UnitQuaternionRotatePoint(camera_tr_rig, point_rig)
  point_out[0] += camera_tr_rig[4];
  point_out[1] += camera_tr_rig[5];
  point_out[2] += camera_tr_rig[6];
  
  return point_out


# 3-Vector dot product:
def DotProduct3(vector1, vector2):
  return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]


def CubicHermiteSpline(p0, p1, p2, p3, x):
  a = (0.5) * (-p0 + (3.0) * p1 - (3.0) * p2 + p3)
  b = (0.5) * ((2.0) * p0 - (5.0) * p1 + (4.0) * p2 - p3)
  c = (0.5) * (-p0 + p2)
  d = p1
  
  return d + x * (c + x * (b + x * a))


def EvalUniformCubicBSpline(a, b, c, d, x):
  # x must be in [3, 4[.
  
  # i == 3
  x_for_d = x - 3
  d_factor = 1./6. * x_for_d * x_for_d * x_for_d
  
  # i == 2
  c_factor = -1./2.*x*x*x + 5*x*x - 16*x + 50./3.
  
  # i == 1
  b_factor = 1./2.*x*x*x - 11./2.*x*x + (39./2.)*x - 131./6.
  
  # i == 0
  a_factor =  -1./6. * (x - 4) * (x - 4) * (x - 4)
  
  return a_factor * a + b_factor * b + c_factor * c + d_factor * d


def NoncentralGenericBicubicModelUnprojection(
    l00, l01, l02, l03, l10, l11, l12, l13, l20, l21, l22, l23, l30, l31, l32, l33,  #camera_intrinsics
    frac_x, frac_y):
  f0 = CubicHermiteSpline(l00, l01, l02, l03, frac_x)
  f1 = CubicHermiteSpline(l10, l11, l12, l13, frac_x)
  f2 = CubicHermiteSpline(l20, l21, l22, l23, frac_x)
  f3 = CubicHermiteSpline(l30, l31, l32, l33, frac_x)
  unprojection = CubicHermiteSpline(f0, f1, f2, f3, frac_y);
  direction = Matrix([[unprojection[0]],
                      [unprojection[1]],
                      [unprojection[2]]])
  direction = direction.normalized()
  
  return Matrix([[direction[0]],
                 [direction[1]],
                 [direction[2]],
                 [unprojection[3]],
                 [unprojection[4]],
                 [unprojection[5]]])


def NoncentralGenericBSplineModelUnprojection(
    l00, l01, l02, l03, l10, l11, l12, l13, l20, l21, l22, l23, l30, l31, l32, l33,  #camera_intrinsics
    frac_x, frac_y):
  f0 = EvalUniformCubicBSpline(l00, l01, l02, l03, frac_x)
  f1 = EvalUniformCubicBSpline(l10, l11, l12, l13, frac_x)
  f2 = EvalUniformCubicBSpline(l20, l21, l22, l23, frac_x)
  f3 = EvalUniformCubicBSpline(l30, l31, l32, l33, frac_x)
  unprojection = EvalUniformCubicBSpline(f0, f1, f2, f3, frac_y);
  direction = Matrix([[unprojection[0]],
                      [unprojection[1]],
                      [unprojection[2]]])
  direction = direction.normalized()
  
  return Matrix([[direction[0]],
                 [direction[1]],
                 [direction[2]],
                 [unprojection[3]],
                 [unprojection[4]],
                 [unprojection[5]]])


def CentralGenericBicubicModelUnprojection(
    p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33,  #camera_intrinsics
    frac_x, frac_y):
  f0 = CubicHermiteSpline(p00, p01, p02, p03, frac_x)
  f1 = CubicHermiteSpline(p10, p11, p12, p13, frac_x)
  f2 = CubicHermiteSpline(p20, p21, p22, p23, frac_x)
  f3 = CubicHermiteSpline(p30, p31, p32, p33, frac_x)
  unprojection = CubicHermiteSpline(f0, f1, f2, f3, frac_y);
  unprojection = unprojection.normalized()
  return Matrix([[unprojection[0]],
                 [unprojection[1]],
                 [unprojection[2]]])


def CentralGenericBicubicModelFittingProblemError(
    p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33,  #camera_intrinsics
    frac_x, frac_y, measurement_x, measurement_y, measurement_z):
  # Interpolation data points:
  #          col
  #      p00 p01 p02 p03
  # row  p10 p11 p12 p13
  #      p20 p21 p22 p23
  #      p30 p31 p32 p33
  f0 = CubicHermiteSpline(p00, p01, p02, p03, frac_x)
  f1 = CubicHermiteSpline(p10, p11, p12, p13, frac_x)
  f2 = CubicHermiteSpline(p20, p21, p22, p23, frac_x)
  f3 = CubicHermiteSpline(p30, p31, p32, p33, frac_x)
  unprojection = CubicHermiteSpline(f0, f1, f2, f3, frac_y);
  unprojection = unprojection.normalized()
  return Matrix([[unprojection[0] - measurement_x],
                 [unprojection[1] - measurement_y],
                 [unprojection[2] - measurement_z]])


def CentralGenericBSplineModelUnprojection(
    p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33,  #camera_intrinsics
    frac_x, frac_y):
  a = EvalUniformCubicBSpline(p00, p01, p02, p03, frac_x)
  b = EvalUniformCubicBSpline(p10, p11, p12, p13, frac_x)
  c = EvalUniformCubicBSpline(p20, p21, p22, p23, frac_x)
  d = EvalUniformCubicBSpline(p30, p31, p32, p33, frac_x)
  unprojection = EvalUniformCubicBSpline(a, b, c, d, frac_y)
  unprojection = unprojection.normalized()
  return Matrix([[unprojection[0]],
                 [unprojection[1]],
                 [unprojection[2]]])


def CentralGenericBSplineModelFittingProblemError(
    p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33,  #camera_intrinsics
    frac_x, frac_y, measurement_x, measurement_y, measurement_z):
  a = EvalUniformCubicBSpline(p00, p01, p02, p03, frac_x)
  b = EvalUniformCubicBSpline(p10, p11, p12, p13, frac_x)
  c = EvalUniformCubicBSpline(p20, p21, p22, p23, frac_x)
  d = EvalUniformCubicBSpline(p30, p31, p32, p33, frac_x)
  unprojection = EvalUniformCubicBSpline(a, b, c, d, frac_y)
  unprojection = unprojection.normalized()
  return Matrix([[unprojection[0] - measurement_x],
                 [unprojection[1] - measurement_y],
                 [unprojection[2] - measurement_z]])


def CentralGenericBilinearModelUnprojection(
    p00, p01, p10, p11,  #camera_intrinsics
    frac_x, frac_y):
  unprojection = ((1 - frac_x) * (1 - frac_y) * p00 +
                  (    frac_x) * (1 - frac_y) * p01 +
                  (1 - frac_x) * (    frac_y) * p10 +
                  (    frac_x) * (    frac_y) * p11)
  unprojection = unprojection.normalized()
  return Matrix([[unprojection[0]],
                 [unprojection[1]],
                 [unprojection[2]]])

def CentralGenericBilinearModelFittingProblemError(
    p00, p01, p10, p11,  #camera_intrinsics
    frac_x, frac_y, measurement_x, measurement_y, measurement_z):
  unprojection = ((1 - frac_x) * (1 - frac_y) * p00 +
                  (    frac_x) * (1 - frac_y) * p01 +
                  (1 - frac_x) * (    frac_y) * p10 +
                  (    frac_x) * (    frac_y) * p11)
  unprojection = unprojection.normalized()
  return Matrix([[unprojection[0] - measurement_x],
                 [unprojection[1] - measurement_y],
                 [unprojection[2] - measurement_z]])


def ConvertDirectionToLocalUpdate(base_direction, target_direction, tangent1, tangent2):
  factor = 1 / DotProduct3(base_direction, target_direction)
  
  offset = (factor * target_direction) - base_direction
  
  return Matrix([[DotProduct3(tangent1, offset)],
                 [DotProduct3(tangent2, offset)]])


# For quaternion layout: (w, x, y, z).
def QuaternionMultiplication(z, w):
  return Matrix([[z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3]],
                 [z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2]],
                 [z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1]],
                 [z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0]]])


# For quaternion layout: (w, x, y, z).
def QuaternionLocalUpdate(delta, q):
  norm_delta = sqrt(delta[0] * delta[0] +
                    delta[1] * delta[1] +
                    delta[2] * delta[2])
  sin_delta_by_delta = sin(norm_delta) / norm_delta
  
  delta_q = Matrix([[cos(norm_delta)],
                    [sin_delta_by_delta * delta[0]],
                    [sin_delta_by_delta * delta[1]],
                    [sin_delta_by_delta * delta[2]]])
  
  return QuaternionMultiplication(delta_q, q)


def ComputeTangentsForLine_ForSmallAbsX(direction):
  other_vector = Matrix([[1], [0], [0]])
  t1 = direction.cross(other_vector).normalized()
  t2 = direction.cross(t1)
  return t1.col_join(t2)

def ComputeTangentsForLine_ForLargeAbsX(direction):
  other_vector = Matrix([[0], [1], [0]])
  t1 = direction.cross(other_vector).normalized()
  t2 = direction.cross(t1)
  return t1.col_join(t2)


def DirectionBorderRegularization(outer, inner1, inner2):
  proj = inner1.dot(inner2) * inner1;
  mirror = proj + (proj - inner2);
  
  return mirror - outer


def CentralThinPrismFisheyeProjection(
    px, py, pz,
    fx, fy, cx, cy,
    k1, k2, k3, k4,
    p1, p2, sx1, sy1,
    fisheye_case):
  nx = px / pz
  ny = py / pz
  
  r = sqrt(nx * nx + ny * ny)
  
  if fisheye_case:
    theta_by_r = atan(r) / r
    fisheye_x = theta_by_r * nx
    fisheye_y = theta_by_r * ny
  else:
    fisheye_x = nx
    fisheye_y = ny
  
  x2 = fisheye_x * fisheye_x
  xy = fisheye_x * fisheye_y
  y2 = fisheye_y * fisheye_y
  r2 = x2 + y2
  r4 = r2 * r2
  r6 = r4 * r2
  r8 = r6 * r2
  
  radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8
  dx = 2 * p1 * xy + p2 * (r2 + 2 * x2) + sx1 * r2
  dy = 2 * p2 * xy + p1 * (r2 + 2 * y2) + sy1 * r2
  
  distorted_x = fisheye_x + radial * fisheye_x + dx
  distorted_y = fisheye_y + radial * fisheye_y + dy
  
  return Matrix([[fx * distorted_x + cx],
                 [fy * distorted_y + cy]])


def CentralOpenCVProjection(
    px, py, pz,
    fx, fy, cx, cy,
    k1, k2, k3, k4,
    k5, k6, p1, p2):
  nx = px / pz
  ny = py / pz
  
  x2 = nx * nx
  xy = nx * ny
  y2 = ny * ny
  r2 = x2 + y2
  r4 = r2 * r2
  r6 = r4 * r2
  
  radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
  dx = 2 * p1 * xy + p2 * (r2 + 2 * x2)
  dy = 2 * p2 * xy + p1 * (r2 + 2 * y2)
  
  distorted_x = nx * radial + dx
  distorted_y = ny * radial + dy
  
  return Matrix([[fx * distorted_x + cx],
                 [fy * distorted_y + cy]])


def CentralRadialProjection(
    spline_resolution, spline_param0, spline_param1, spline_param2, spline_param3,
    fx, fy, cx, cy, p1, p2, sx1, sy1,
    lx, ly, lz):
  local_point = Matrix([[lx],
                        [ly],
                        [lz]])
  
  # Radial part
  original_angle = acos(local_point.normalized()[2]);
  
  pos_in_spline = 1. + (spline_resolution - 3.) / (math.pi / 2) * original_angle;
  # chunk = std::max(1, std::min(spline_resolution() - 3, static_cast<int>(pos_in_spline)));
  fraction = frac(pos_in_spline) # - chunk;
  
  radial_factor = EvalUniformCubicBSpline(
      spline_param0,
      spline_param1,
      spline_param2,
      spline_param3,
      fraction + 3.);
  
  # Parametric part
  nx = lx / lz
  ny = ly / lz
  
  x2 = nx * nx
  xy = nx * ny
  y2 = ny * ny
  r2 = x2 + y2
  
  dx = 2 * p1 * xy + p2 * (r2 + 2 * x2) + sx1 * r2
  dy = 2 * p2 * xy + p1 * (r2 + 2 * y2) + sy1 * r2
  
  distorted_x = nx + radial_factor * nx + dx
  distorted_y = ny + radial_factor * ny + dy
  
  return Matrix([[fx * distorted_x + cx],
                 [fy * distorted_y + cy]])


if __name__ == '__main__':
  p00 = Matrix(3, 1, lambda i,j:Symbol('p00_%d' % (i), real=True))
  p01 = Matrix(3, 1, lambda i,j:Symbol('p01_%d' % (i), real=True))
  p02 = Matrix(3, 1, lambda i,j:Symbol('p02_%d' % (i), real=True))
  p03 = Matrix(3, 1, lambda i,j:Symbol('p03_%d' % (i), real=True))
  p10 = Matrix(3, 1, lambda i,j:Symbol('p10_%d' % (i), real=True))
  p11 = Matrix(3, 1, lambda i,j:Symbol('p11_%d' % (i), real=True))
  p12 = Matrix(3, 1, lambda i,j:Symbol('p12_%d' % (i), real=True))
  p13 = Matrix(3, 1, lambda i,j:Symbol('p13_%d' % (i), real=True))
  p20 = Matrix(3, 1, lambda i,j:Symbol('p20_%d' % (i), real=True))
  p21 = Matrix(3, 1, lambda i,j:Symbol('p21_%d' % (i), real=True))
  p22 = Matrix(3, 1, lambda i,j:Symbol('p22_%d' % (i), real=True))
  p23 = Matrix(3, 1, lambda i,j:Symbol('p23_%d' % (i), real=True))
  p30 = Matrix(3, 1, lambda i,j:Symbol('p30_%d' % (i), real=True))
  p31 = Matrix(3, 1, lambda i,j:Symbol('p31_%d' % (i), real=True))
  p32 = Matrix(3, 1, lambda i,j:Symbol('p32_%d' % (i), real=True))
  p33 = Matrix(3, 1, lambda i,j:Symbol('p33_%d' % (i), real=True))
  
  l00 = Matrix(6, 1, lambda i,j:Symbol('l00_%d' % (i), real=True))
  l01 = Matrix(6, 1, lambda i,j:Symbol('l01_%d' % (i), real=True))
  l02 = Matrix(6, 1, lambda i,j:Symbol('l02_%d' % (i), real=True))
  l03 = Matrix(6, 1, lambda i,j:Symbol('l03_%d' % (i), real=True))
  l10 = Matrix(6, 1, lambda i,j:Symbol('l10_%d' % (i), real=True))
  l11 = Matrix(6, 1, lambda i,j:Symbol('l11_%d' % (i), real=True))
  l12 = Matrix(6, 1, lambda i,j:Symbol('l12_%d' % (i), real=True))
  l13 = Matrix(6, 1, lambda i,j:Symbol('l13_%d' % (i), real=True))
  l20 = Matrix(6, 1, lambda i,j:Symbol('l20_%d' % (i), real=True))
  l21 = Matrix(6, 1, lambda i,j:Symbol('l21_%d' % (i), real=True))
  l22 = Matrix(6, 1, lambda i,j:Symbol('l22_%d' % (i), real=True))
  l23 = Matrix(6, 1, lambda i,j:Symbol('l23_%d' % (i), real=True))
  l30 = Matrix(6, 1, lambda i,j:Symbol('l30_%d' % (i), real=True))
  l31 = Matrix(6, 1, lambda i,j:Symbol('l31_%d' % (i), real=True))
  l32 = Matrix(6, 1, lambda i,j:Symbol('l32_%d' % (i), real=True))
  l33 = Matrix(6, 1, lambda i,j:Symbol('l33_%d' % (i), real=True))
  
  frac_x = Symbol("frac_x", real=True)
  frac_y = Symbol("frac_y", real=True)
  measurement_x = Symbol("measurement_x", real=True)
  measurement_y = Symbol("measurement_y", real=True)
  measurement_z = Symbol("measurement_z", real=True)
  
  
  # For pose and geometry optimization:
  # Local point Jacobian wrt. image_tr_global, pattern_point
  image_tr_global = Matrix(7, 1, lambda i,j:Symbol('itg_%d' % (i), real=True))
  pattern_point = Matrix(3, 1, lambda i,j:Symbol('p_%d' % (i), real=True))
  
  parameters = image_tr_global.col_join(pattern_point)
  functions = [lambda variables : TransformPoint(variables.extract([0, 1, 2, 3, 4, 5, 6], [0]), variables.extract([7, 8, 9], [0]))]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=True, simplify_residual=False)
  
  # For rig pose and geometry optimization:
  # Local point Jacobian wrt. camera_tr_rig, rig_tr_global, pattern_point
  camera_tr_rig = Matrix(7, 1, lambda i,j:Symbol('ctr_%d' % (i), real=True))
  rig_tr_global = Matrix(7, 1, lambda i,j:Symbol('rtg_%d' % (i), real=True))
  pattern_point = Matrix(3, 1, lambda i,j:Symbol('p_%d' % (i), real=True))
  
  parameters = rig_tr_global.col_join(camera_tr_rig).col_join(pattern_point)
  functions = [lambda variables : RigTransformPoint(
      variables.extract([7, 8, 9, 10, 11, 12, 13], [0]),
      variables.extract([0, 1, 2, 3, 4, 5, 6], [0]),
      variables.extract([14, 15, 16], [0]))]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=True, simplify_residual=False)
  
  # Tangents Jacobian wrt. direction:
  direction = Matrix(3, 1, lambda i,j:Symbol('dir_%d' % (i), real=True))
  OptimizerBuilder([lambda variables : ComputeTangentsForLine_ForSmallAbsX(variables)],
                   direction,
                   direction,
                   simplify_function_jacobian=[True],
                   simplify_jacobian=True, simplify_residual=True)
  OptimizerBuilder([lambda variables : ComputeTangentsForLine_ForLargeAbsX(variables)],
                   direction,
                   direction,
                   simplify_function_jacobian=[True],
                   simplify_jacobian=True, simplify_residual=True)
  
  
  # Jacobian for CentralGenericBilinear unprojection wrt. pixel x, y
  # (CentralGenericBilinear_UnprojectFromPixelCornerConv_ComputeResidualAndJacobian()):
  parameters = Matrix([[frac_x],
                       [frac_y]])
  
  functions = [lambda variables : CentralGenericBilinearModelUnprojection(
                                      p00, p01, p10, p11,
                                      variables[0], variables[1])]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # CentralGenericBilinearDirectionCostFunction_ComputeResidualAndJacobian():
  # Residual: grid.InterpolateBilinearVector(model->PixelCornerConvToGridPoint(x + 0.5f, y + 0.5f)) - measurement
  # Variables are p00 .. p33
  parameters = p00.col_join(
                 p01.col_join(
                   p10.col_join(
                     p11)))
  functions = [lambda variables : CentralGenericBilinearModelFittingProblemError(
                  variables.extract([0, 1, 2], [0]),
                  variables.extract([3, 4, 5], [0]),
                  variables.extract([6, 7, 8], [0]),
                  variables.extract([9, 10, 11], [0]),
                  frac_x, frac_y, measurement_x, measurement_y, measurement_z)]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # CentralGenericBSplineDirectionCostFunction_ComputeResidualAndJacobian():
  # Residual: grid.InterpolateBSplineVector(model->PixelCornerConvToGridPoint(x + 0.5f, y + 0.5f)) - measurement
  # Variables are p00 .. p33
  parameters = p00.col_join(
                 p01.col_join(
                   p02.col_join(
                     p03.col_join(
                       p10.col_join(
                         p11.col_join(
                           p12.col_join(
                             p13.col_join(
                               p20.col_join(
                                 p21.col_join(
                                   p22.col_join(
                                     p23.col_join(
                                       p30.col_join(
                                         p31.col_join(
                                           p32.col_join(
                                             p33)))))))))))))))
  functions = [lambda variables : CentralGenericBSplineModelFittingProblemError(
                  variables.extract([0, 1, 2], [0]),
                  variables.extract([3, 4, 5], [0]),
                  variables.extract([6, 7, 8], [0]),
                  variables.extract([9, 10, 11], [0]),
                  variables.extract([12, 13, 14], [0]),
                  variables.extract([15, 16, 17], [0]),
                  variables.extract([18, 19, 20], [0]),
                  variables.extract([21, 22, 23], [0]),
                  variables.extract([24, 25, 26], [0]),
                  variables.extract([27, 28, 29], [0]),
                  variables.extract([30, 31, 32], [0]),
                  variables.extract([33, 34, 35], [0]),
                  variables.extract([36, 37, 38], [0]),
                  variables.extract([39, 40, 41], [0]),
                  variables.extract([42, 43, 44], [0]),
                  variables.extract([45, 46, 47], [0]),
                  frac_x, frac_y, measurement_x, measurement_y, measurement_z)]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian for CentralGenericBSpline unprojection wrt. pixel x, y
  # (CentralGenericBSpline_UnprojectFromPixelCornerConv_ComputeResidualAndJacobian()):
  parameters = Matrix([[frac_x],
                       [frac_y]])
  
  functions = [lambda variables : CentralGenericBSplineModelUnprojection(
                                      p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33,
                                      variables[0], variables[1])]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian for direction grid border regularization:
  outer = Matrix(3, 1, lambda i,j:Symbol('o_%d' % (i), real=True))
  inner1 = Matrix(3, 1, lambda i,j:Symbol('i1_%d' % (i), real=True))
  inner2 = Matrix(3, 1, lambda i,j:Symbol('i2_%d' % (i), real=True))
  parameters = outer.col_join(inner1.col_join(inner2))
  OptimizerBuilder([lambda variables : DirectionBorderRegularization(
                        variables.extract([0, 1, 2], [0]),
                        variables.extract([3, 4, 5], [0]),
                        variables.extract([6, 7, 8], [0]))],
                   parameters,
                   parameters,
                   simplify_function_jacobian=[True],
                   simplify_jacobian=True, simplify_residual=True)
  
  
  # Derive Jacobian of local update to quaternions (as in ceres)
  # TODO: This only works if replacing subs() by limit() in optimizer_builder's
  #       ComputeValueAndJacobian(). However, it seems that this gave wrong results in other cases ...
  q = Matrix(4, 1, lambda i,j:Symbol('q_%d' % (i), real=True))
  delta_q = Matrix(3, 1, lambda i,j:Symbol('dq_%d' % (i), real=True))
  OptimizerBuilder([lambda variables : QuaternionLocalUpdate(variables, q)],
                   delta_q,
                   Matrix([[0], [0], [0]]),
                   simplify_function_jacobian=[True],
                   simplify_jacobian=True, simplify_residual=True)
  
  
  # Derivation of LocalUpdateJacobianWrtDirection():
  target_direction = Matrix(3, 1, lambda i,j:Symbol('t_%d' % (i), real=True))
  base_direction = Matrix(3, 1, lambda i,j:Symbol('d_%d' % (i), real=True))
  tangent1 = Matrix(3, 1, lambda i,j:Symbol('t1_%d' % (i), real=True))
  tangent2 = Matrix(3, 1, lambda i,j:Symbol('t2_%d' % (i), real=True))
  
  parameters = target_direction
  parameter_values = base_direction  # Taking Jacobian at base_direction
  
  functions = [lambda target_dir : ConvertDirectionToLocalUpdate(base_direction, target_dir, tangent1, tangent2)]
  
  OptimizerBuilder(functions,
                   parameters, parameter_values,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian for NoncentralGenericBicubic unprojection wrt. pixel x, y
  # (NoncentralGenericBicubic_UnprojectFromPixelCornerConv_ComputeResidualAndJacobian()):
  parameters = Matrix([[frac_x],
                       [frac_y]])
  
  functions = [lambda variables : NoncentralGenericBicubicModelUnprojection(
                                      l00, l01, l02, l03, l10, l11, l12, l13, l20, l21, l22, l23, l30, l31, l32, l33,
                                      variables[0], variables[1])]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian for CentralGenericBicubic unprojection wrt. pixel x, y
  # (CentralGenericBicubic_UnprojectFromPixelCornerConv_ComputeResidualAndJacobian()):
  parameters = Matrix([[frac_x],
                       [frac_y]])
  
  functions = [lambda variables : CentralGenericBicubicModelUnprojection(
                                      p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p33,
                                      variables[0], variables[1])]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # CentralGenericBicubicDirectionCostFunction_ComputeResidualAndJacobian():
  # Residual: grid.InterpolateBicubicVector(model->PixelCornerConvToGridPoint(x + 0.5f, y + 0.5f)) - measurement
  # Variables are p00 .. p33
  parameters = p00.col_join(
                 p01.col_join(
                   p02.col_join(
                     p03.col_join(
                       p10.col_join(
                         p11.col_join(
                           p12.col_join(
                             p13.col_join(
                               p20.col_join(
                                 p21.col_join(
                                   p22.col_join(
                                     p23.col_join(
                                       p30.col_join(
                                         p31.col_join(
                                           p32.col_join(
                                             p33)))))))))))))))
  functions = [lambda variables : CentralGenericBicubicModelFittingProblemError(
                  variables.extract([0, 1, 2], [0]),
                  variables.extract([3, 4, 5], [0]),
                  variables.extract([6, 7, 8], [0]),
                  variables.extract([9, 10, 11], [0]),
                  variables.extract([12, 13, 14], [0]),
                  variables.extract([15, 16, 17], [0]),
                  variables.extract([18, 19, 20], [0]),
                  variables.extract([21, 22, 23], [0]),
                  variables.extract([24, 25, 26], [0]),
                  variables.extract([27, 28, 29], [0]),
                  variables.extract([30, 31, 32], [0]),
                  variables.extract([33, 34, 35], [0]),
                  variables.extract([36, 37, 38], [0]),
                  variables.extract([39, 40, 41], [0]),
                  variables.extract([42, 43, 44], [0]),
                  variables.extract([45, 46, 47], [0]),
                  frac_x, frac_y, measurement_x, measurement_y, measurement_z)]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian for NoncentralGenericBSpline unprojection wrt. pixel x, y
  # (NoncentralGenericBicubic_UnprojectFromPixelCornerConv_ComputeResidualAndJacobian()):
  parameters = Matrix([[frac_x],
                       [frac_y]])
  
  functions = [lambda variables : NoncentralGenericBSplineModelUnprojection(
                                      l00, l01, l02, l03, l10, l11, l12, l13, l20, l21, l22, l23, l30, l31, l32, l33,
                                      variables[0], variables[1])]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian for CentralThinPrismFisheyeModel::ProjectPointToPixelCornerConv() wrt. the 12 camera model parameters.
  fx = Symbol("fx", real=True)
  fy = Symbol("fy", real=True)
  cx = Symbol("cx", real=True)
  cy = Symbol("cy", real=True)
  k1 = Symbol("k1", real=True)
  k2 = Symbol("k2", real=True)
  k3 = Symbol("k3", real=True)
  k4 = Symbol("k4", real=True)
  p1 = Symbol("p1", real=True)
  p2 = Symbol("p2", real=True)
  sx1 = Symbol("sx1", real=True)
  sy1 = Symbol("sy1", real=True)
  local_point = Matrix(3, 1, lambda i,j:Symbol('p_%d' % (i), real=True))
  
  parameters = Matrix([[fx],
                       [fy],
                       [cx],
                       [cy],
                       [k1],
                       [k2],
                       [k3],
                       [k4],
                       [p1],
                       [p2],
                       [sx1],
                       [sy1]])
  
  print('Fisheye case:')
  functions = [lambda variables : CentralThinPrismFisheyeProjection(
                                      local_point[0], local_point[1], local_point[2],
                                      variables[0], variables[1], variables[2], variables[3],
                                      variables[4], variables[5], variables[6], variables[7],
                                      variables[8], variables[9], variables[10], variables[11], True)]
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  print('Non-fisheye case:')
  functions = [lambda variables : CentralThinPrismFisheyeProjection(
                                      local_point[0], local_point[1], local_point[2],
                                      variables[0], variables[1], variables[2], variables[3],
                                      variables[4], variables[5], variables[6], variables[7],
                                      variables[8], variables[9], variables[10], variables[11], False)]
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian for CentralOpenCVModel::ProjectPointToPixelCornerConv() wrt. the 12 camera model parameters.
  fx = Symbol("fx", real=True)
  fy = Symbol("fy", real=True)
  cx = Symbol("cx", real=True)
  cy = Symbol("cy", real=True)
  k1 = Symbol("k1", real=True)
  k2 = Symbol("k2", real=True)
  k3 = Symbol("k3", real=True)
  k4 = Symbol("k4", real=True)
  k5 = Symbol("k5", real=True)
  k6 = Symbol("k6", real=True)
  p1 = Symbol("p1", real=True)
  p2 = Symbol("p2", real=True)
  local_point = Matrix(3, 1, lambda i,j:Symbol('p_%d' % (i), real=True))
  
  parameters = Matrix([[fx],
                       [fy],
                       [cx],
                       [cy],
                       [k1],
                       [k2],
                       [k3],
                       [k4],
                       [k5],
                       [k6],
                       [p1],
                       [p2]])
  
  functions = [lambda variables : CentralOpenCVProjection(
                                      local_point[0], local_point[1], local_point[2],
                                      variables[0], variables[1], variables[2], variables[3],
                                      variables[4], variables[5], variables[6], variables[7],
                                      variables[8], variables[9], variables[10], variables[11])]
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian of CentralRadialModel::Project() wrt. the local point.
  fx = Symbol("fx", real=True)
  fy = Symbol("fy", real=True)
  cx = Symbol("cx", real=True)
  cy = Symbol("cy", real=True)
  p1 = Symbol("p1", real=True)
  p2 = Symbol("p2", real=True)
  sx1 = Symbol("sx1", real=True)
  sy1 = Symbol("sy1", real=True)
  spline_resolution = Symbol("spline_resolution", real=True)
  spline_param0 = Symbol("spline_param0", real=True)
  spline_param1 = Symbol("spline_param1", real=True)
  spline_param2 = Symbol("spline_param2", real=True)
  spline_param3 = Symbol("spline_param3", real=True)
  
  local_point = Matrix(3, 1, lambda i,j:Symbol('p_%d' % (i), real=True))
  
  parameters = Matrix([[local_point[0]],
                       [local_point[1]],
                       [local_point[2]]])
  
  functions = [lambda variables : CentralRadialProjection(
                                      spline_resolution, spline_param0, spline_param1, spline_param2, spline_param3,
                                      fx, fy, cx, cy, p1, p2, sx1, sy1,
                                      variables[0], variables[1], variables[2])]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
  
  # Jacobian of CentralRadialModel::Project() wrt. the camera model parameters.
  fx = Symbol("fx", real=True)
  fy = Symbol("fy", real=True)
  cx = Symbol("cx", real=True)
  cy = Symbol("cy", real=True)
  p1 = Symbol("p1", real=True)
  p2 = Symbol("p2", real=True)
  sx1 = Symbol("sx1", real=True)
  sy1 = Symbol("sy1", real=True)
  spline_resolution = Symbol("spline_resolution", real=True)
  spline_param0 = Symbol("spline_param0", real=True)
  spline_param1 = Symbol("spline_param1", real=True)
  spline_param2 = Symbol("spline_param2", real=True)
  spline_param3 = Symbol("spline_param3", real=True)
  
  local_point = Matrix(3, 1, lambda i,j:Symbol('p_%d' % (i), real=True))
  
  parameters = Matrix([[fx],
                       [fy],
                       [cx],
                       [cy],
                       [p1],
                       [p2],
                       [sx1],
                       [sy1],
                       [spline_param0],
                       [spline_param1],
                       [spline_param2],
                       [spline_param3]])
  
  functions = [lambda variables : CentralRadialProjection(
                                      spline_resolution, variables[8], variables[9], variables[10], variables[11],
                                      variables[0], variables[1], variables[2], variables[3],
                                      variables[4], variables[5], variables[6], variables[7],
                                      local_point[0], local_point[1], local_point[2])]
  
  OptimizerBuilder(functions,
                   parameters, parameters,
                   simplify_function_jacobian=[False],
                   simplify_jacobian=False, simplify_residual=False)
  
