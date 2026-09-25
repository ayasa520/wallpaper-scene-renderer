#pragma once

#include <string_view>

namespace wallpaper {

// Install after the vector and Mat3 preludes, before any callback or native matrix result.
// One constructor per realm gives authored and native matrices the same value identity.
// Each matrix owns its column-major array; arithmetic creates independent storage, while
// translation(Vec2/Vec3) is the explicit mutating accessor used for chained assignments.
inline constexpr std::string_view kSceneScriptMat4Prelude = R"JS(
  const Mat4 = (globalThis.Mat4 = class Mat4 {
    constructor(value) {
      if (value instanceof Mat4) this.m = value.m.slice();
      else if (Array.isArray(value) && value.length === 16) this.m = value.slice();
      else {
        const parsed = typeof value === 'string' ? value.split(' ').map(parseFloat) : [];
        this.m = parsed.length === 16 ? parsed : [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
      }
    }
    static identity() { return new Mat4(); }
    static fromTranslation(position) {
      const result = new Mat4();
      if (position instanceof Vec3) {
        result.m[12] = position.x; result.m[13] = position.y; result.m[14] = position.z;
      } else if (position instanceof Vec2) {
        result.m[12] = position.x; result.m[13] = position.y;
      }
      return result;
    }
    static fromScale(scale) {
      const result = new Mat4();
      if (typeof scale === 'number') {
        result.m[0] = scale; result.m[5] = scale; result.m[10] = scale;
      } else {
        result.m[0] = scale.x; result.m[5] = scale.y; result.m[10] = scale.z;
      }
      return result;
    }
    static fromRotation(angle, axis) {
      const radians = angle * (Math.PI / 180);
      const direction = axis.normalize();
      const c = Math.cos(radians), s = Math.sin(radians), t = 1 - c;
      const x = direction.x, y = direction.y, z = direction.z;
      const result = new Mat4();
      result.m[0] = c + x*x*t; result.m[1] = x*y*t + z*s; result.m[2] = x*z*t - y*s;
      result.m[4] = y*x*t - z*s; result.m[5] = c + y*y*t; result.m[6] = y*z*t + x*s;
      result.m[8] = z*x*t + y*s; result.m[9] = z*y*t - x*s; result.m[10] = c + z*z*t;
      return result;
    }
    static fromEuler(x, y, z) {
      if (x instanceof Vec3) { z = x.z; y = x.y; x = x.x; }
      // Euler inputs are degrees. With column vectors the composed rotation is Rz * Ry * Rx;
      // writing its basis directly preserves this order for both numeric and Vec3 inputs.
      const rx = x * (Math.PI / 180), ry = y * (Math.PI / 180), rz = z * (Math.PI / 180);
      const c1 = Math.cos(-rz), s1 = Math.sin(-rz);
      const c2 = Math.cos(-ry), s2 = Math.sin(-ry);
      const c3 = Math.cos(-rx), s3 = Math.sin(-rx);
      const result = new Mat4();
      result.m[0] = c1*c2; result.m[1] = -c2*s1; result.m[2] = s2;
      result.m[4] = c3*s1 + c1*s2*s3;
      result.m[5] = c1*c3 - s1*s2*s3;
      result.m[6] = -c2*s3;
      result.m[8] = s1*s3 - c1*c3*s2;
      result.m[9] = c1*s3 + c3*s1*s2;
      result.m[10] = c2*c3;
      return result;
    }
    static fromBasis(right, up, forward) {
      const result = new Mat4();
      result.m[0] = right.x; result.m[1] = right.y; result.m[2] = right.z;
      result.m[4] = up.x; result.m[5] = up.y; result.m[6] = up.z;
      result.m[8] = forward.x; result.m[9] = forward.y; result.m[10] = forward.z;
      return result;
    }
    static lookAt(eye, center, up) {
      const forward = center.subtract(eye).normalize();
      const right = forward.cross(up).normalize();
      const vertical = right.cross(forward);
      const result = new Mat4();
      // A view transform stores the camera axes as rows, then translates the eye to the
      // origin in that basis. fromBasis instead supplies columns for a model transform.
      result.m[0] = right.x; result.m[1] = vertical.x; result.m[2] = -forward.x;
      result.m[4] = right.y; result.m[5] = vertical.y; result.m[6] = -forward.y;
      result.m[8] = right.z; result.m[9] = vertical.z; result.m[10] = -forward.z;
      result.m[12] = -right.dot(eye);
      result.m[13] = -vertical.dot(eye);
      result.m[14] = forward.dot(eye);
      return result;
    }
    static compose(translation, rotation, scale) {
      return Mat4.fromTranslation(translation).multiply(Mat4.fromEuler(rotation)).scale(scale);
    }
    translation(position) {
      if (position instanceof Vec3 || position instanceof Vec2) {
        this.m[12] = position.x;
        this.m[13] = position.y;
        this.m[14] = position instanceof Vec3 ? position.z : 0;
        return this;
      }
      return new Vec3(this.m[12], this.m[13], this.m[14]);
    }
    right() { return new Vec3(this.m[0], this.m[1], this.m[2]); }
    up() { return new Vec3(this.m[4], this.m[5], this.m[6]); }
    forward() { return new Vec3(this.m[8], this.m[9], this.m[10]); }
    add(other) {
      const result = new Mat4();
      for (let index = 0; index < 16; ++index) result.m[index] = this.m[index] + other.m[index];
      return result;
    }
    subtract(other) {
      const result = new Mat4();
      for (let index = 0; index < 16; ++index) result.m[index] = this.m[index] - other.m[index];
      return result;
    }
    multiply(other) {
      if (other instanceof Mat4) {
        const a = this.m, b = other.m;
        const result = new Mat4();
        // Accumulate a row of the left operand against a column of the right operand.
        // Separate result storage also makes a.multiply(a) leave its source unchanged.
        for (let column = 0; column < 4; ++column) {
          for (let row = 0; row < 4; ++row) {
            let value = 0;
            for (let inner = 0; inner < 4; ++inner) {
              value += a[inner * 4 + row] * b[column * 4 + inner];
            }
            result.m[column * 4 + row] = value;
          }
        }
        return result;
      } else if (other instanceof Vec4) {
        const m = this.m;
        return new Vec4(m[0]*other.x + m[4]*other.y + m[8]*other.z + m[12]*other.w,
                        m[1]*other.x + m[5]*other.y + m[9]*other.z + m[13]*other.w,
                        m[2]*other.x + m[6]*other.y + m[10]*other.z + m[14]*other.w,
                        m[3]*other.x + m[7]*other.y + m[11]*other.z + m[15]*other.w);
      } else if (typeof other === 'number') {
        const result = new Mat4();
        for (let index = 0; index < 16; ++index) result.m[index] = this.m[index] * other;
        return result;
      }
    }
    translate(position) {
      const m = this.m, x = position.x, y = position.y;
      const z = position instanceof Vec2 ? 0 : position.z;
      const result = new Mat4(this);
      // Right multiplication changes the last column, including its homogeneous entry.
      // Keep the full fourth row for authored non-affine matrices as well as layer TRS.
      result.m[12] = m[0]*x + m[4]*y + m[8]*z + m[12];
      result.m[13] = m[1]*x + m[5]*y + m[9]*z + m[13];
      result.m[14] = m[2]*x + m[6]*y + m[10]*z + m[14];
      result.m[15] = m[3]*x + m[7]*y + m[11]*z + m[15];
      return result;
    }
    rotate(angle, axis) { return this.multiply(Mat4.fromRotation(angle, axis)); }
    scale(scale) {
      const m = this.m;
      const x = typeof scale === 'number' ? scale : scale.x;
      const y = typeof scale === 'number' ? scale : scale.y;
      const z = typeof scale === 'number' ? scale : scale.z;
      const result = new Mat4();
      result.m[0] = m[0]*x; result.m[1] = m[1]*x; result.m[2] = m[2]*x; result.m[3] = m[3]*x;
      result.m[4] = m[4]*y; result.m[5] = m[5]*y; result.m[6] = m[6]*y; result.m[7] = m[7]*y;
      result.m[8] = m[8]*z; result.m[9] = m[9]*z; result.m[10] = m[10]*z; result.m[11] = m[11]*z;
      result.m[12] = m[12]; result.m[13] = m[13]; result.m[14] = m[14]; result.m[15] = m[15];
      return result;
    }
    // Point and direction operations return the first three homogeneous components with
    // w=1 and w=0 respectively. Neither performs a perspective divide, even for a matrix
    // whose fourth row changes the resulting w component.
    transformPoint(point) {
      const m = this.m;
      return new Vec3(m[0]*point.x + m[4]*point.y + m[8]*point.z + m[12],
                      m[1]*point.x + m[5]*point.y + m[9]*point.z + m[13],
                      m[2]*point.x + m[6]*point.y + m[10]*point.z + m[14]);
    }
    transformDirection(direction) {
      const m = this.m;
      return new Vec3(m[0]*direction.x + m[4]*direction.y + m[8]*direction.z,
                      m[1]*direction.x + m[5]*direction.y + m[9]*direction.z,
                      m[2]*direction.x + m[6]*direction.y + m[10]*direction.z);
    }
    transpose() {
      const result = new Mat4();
      for (let column = 0; column < 4; ++column) {
        for (let row = 0; row < 4; ++row) result.m[column * 4 + row] = this.m[row * 4 + column];
      }
      return result;
    }
    inverse() {
      // Build the complete adjugate from reusable two-by-two minors. Retaining all sixteen
      // entries supports non-affine inputs; dividing by the determinant keeps JavaScript's
      // ordinary numerical behavior when that determinant is zero.
      const m = this.m;
      const c00 = m[10]*m[15] - m[14]*m[11];
      const c02 = m[6]*m[15] - m[14]*m[7];
      const c03 = m[6]*m[11] - m[10]*m[7];
      const c04 = m[9]*m[15] - m[13]*m[11];
      const c06 = m[5]*m[15] - m[13]*m[7];
      const c07 = m[5]*m[11] - m[9]*m[7];
      const c08 = m[9]*m[14] - m[13]*m[10];
      const c10 = m[5]*m[14] - m[13]*m[6];
      const c11 = m[5]*m[10] - m[9]*m[6];
      const c12 = m[8]*m[15] - m[12]*m[11];
      const c14 = m[4]*m[15] - m[12]*m[7];
      const c15 = m[4]*m[11] - m[8]*m[7];
      const c16 = m[8]*m[14] - m[12]*m[10];
      const c18 = m[4]*m[14] - m[12]*m[6];
      const c19 = m[4]*m[10] - m[8]*m[6];
      const c20 = m[8]*m[13] - m[12]*m[9];
      const c22 = m[4]*m[13] - m[12]*m[5];
      const c23 = m[4]*m[9] - m[8]*m[5];
      const result = new Mat4();
      result.m[0] = m[5]*c00 - m[6]*c04 + m[7]*c08;
      result.m[1] = -(m[1]*c00 - m[2]*c04 + m[3]*c08);
      result.m[2] = m[1]*c02 - m[2]*c06 + m[3]*c10;
      result.m[3] = -(m[1]*c03 - m[2]*c07 + m[3]*c11);
      result.m[4] = -(m[4]*c00 - m[6]*c12 + m[7]*c16);
      result.m[5] = m[0]*c00 - m[2]*c12 + m[3]*c16;
      result.m[6] = -(m[0]*c02 - m[2]*c14 + m[3]*c18);
      result.m[7] = m[0]*c03 - m[2]*c15 + m[3]*c19;
      result.m[8] = m[4]*c04 - m[5]*c12 + m[7]*c20;
      result.m[9] = -(m[0]*c04 - m[1]*c12 + m[3]*c20);
      result.m[10] = m[0]*c06 - m[1]*c14 + m[3]*c22;
      result.m[11] = -(m[0]*c07 - m[1]*c15 + m[3]*c23);
      result.m[12] = -(m[4]*c08 - m[5]*c16 + m[6]*c20);
      result.m[13] = m[0]*c08 - m[1]*c16 + m[2]*c20;
      result.m[14] = -(m[0]*c10 - m[1]*c18 + m[2]*c22);
      result.m[15] = m[0]*c11 - m[1]*c19 + m[2]*c23;
      const determinant = m[0]*result.m[0] + m[1]*result.m[4] + m[2]*result.m[8] + m[3]*result.m[12];
      const reciprocal = 1 / determinant;
      for (let index = 0; index < 16; ++index) result.m[index] *= reciprocal;
      return result;
    }
    determinant() {
      const m = this.m;
      const c00 = m[10]*m[15] - m[14]*m[11];
      const c04 = m[9]*m[15] - m[13]*m[11];
      const c08 = m[9]*m[14] - m[13]*m[10];
      const c12 = m[8]*m[15] - m[12]*m[11];
      const c16 = m[8]*m[14] - m[12]*m[10];
      const c20 = m[8]*m[13] - m[12]*m[9];
      const i0 = m[5]*c00 - m[6]*c04 + m[7]*c08;
      const i1 = -(m[4]*c00 - m[6]*c12 + m[7]*c16);
      const i2 = m[4]*c04 - m[5]*c12 + m[7]*c20;
      const i3 = -(m[4]*c08 - m[5]*c16 + m[6]*c20);
      return m[0]*i0 + m[1]*i1 + m[2]*i2 + m[3]*i3;
    }
    extractEuler() {
      const m = this.m;
      const z = Math.atan2(m[1], m[0]);
      const c2 = Math.sqrt(m[6]*m[6] + m[10]*m[10]);
      const y = Math.atan2(-m[2], c2);
      const s1 = Math.sin(z), c1 = Math.cos(z);
      const x = Math.atan2(s1*m[8] - c1*m[9], c1*m[5] - s1*m[4]);
      return new Vec3(x * (180 / Math.PI), y * (180 / Math.PI), z * (180 / Math.PI));
    }
    normalMatrix() { return Mat3.fromMat4(this).inverse().transpose(); }
    decompose() {
      const m = this.m;
      const translation = new Vec3(m[12], m[13], m[14]);
      let x = Math.sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
      const y = Math.sqrt(m[4]*m[4] + m[5]*m[5] + m[6]*m[6]);
      const z = Math.sqrt(m[8]*m[8] + m[9]*m[9] + m[10]*m[10]);
      const determinant = m[0]*(m[5]*m[10] - m[9]*m[6])
                        - m[4]*(m[1]*m[10] - m[9]*m[2])
                        + m[8]*(m[1]*m[6] - m[5]*m[2]);
      // Carry a reflection on signed X scale before extracting the rotation. Removing each
      // nonzero column's scale recovers a rotation basis; a collapsed column remains zero.
      // All decomposition members are separate vectors and the source matrix stays intact.
      if (determinant < 0) x = -x;
      const scale = new Vec3(x, y, z);
      const rotationMatrix = new Mat4(this);
      if (x !== 0) { rotationMatrix.m[0] /= x; rotationMatrix.m[1] /= x; rotationMatrix.m[2] /= x; }
      if (y !== 0) { rotationMatrix.m[4] /= y; rotationMatrix.m[5] /= y; rotationMatrix.m[6] /= y; }
      if (z !== 0) { rotationMatrix.m[8] /= z; rotationMatrix.m[9] /= z; rotationMatrix.m[10] /= z; }
      rotationMatrix.m[12] = 0; rotationMatrix.m[13] = 0; rotationMatrix.m[14] = 0;
      return { translation, rotation: rotationMatrix.extractEuler(), scale };
    }
    copy() { return new Mat4(this); }
    equals(other) {
      if (!(other instanceof Mat4)) return false;
      for (let index = 0; index < 16; ++index) {
        if (Math.abs(this.m[index] - other.m[index]) >= __vectorEpsilon) return false;
      }
      return true;
    }
    toString() {
      return this.m[0]+' '+this.m[1]+' '+this.m[2]+' '+this.m[3]+' '+
             this.m[4]+' '+this.m[5]+' '+this.m[6]+' '+this.m[7]+' '+
             this.m[8]+' '+this.m[9]+' '+this.m[10]+' '+this.m[11]+' '+
             this.m[12]+' '+this.m[13]+' '+this.m[14]+' '+this.m[15];
    }
    toConfigString() { return this.toString(); }
  });
)JS";

} // namespace wallpaper
