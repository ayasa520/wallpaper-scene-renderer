#pragma once

#include <string_view>

namespace wallpaper {

// Install after the vector prelude in each JavaScript realm. Native attachment results and
// authored matrices use the same constructor, so instanceof checks and chained operations
// preserve their value types across callbacks. The matrix owns a column-major array: all
// arithmetic returns new storage, while translation(Vec2) is the explicit mutating accessor.
inline constexpr std::string_view kSceneScriptMat3Prelude = R"JS(
  const Mat3 = (globalThis.Mat3 ??= class Mat3 {
    constructor(value) {
      if (value instanceof Mat3) this.m = value.m.slice();
      else if (Array.isArray(value) && value.length === 9) this.m = value.slice();
      else {
        const parsed = typeof value === 'string' ? value.split(' ').map(parseFloat) : [];
        this.m = parsed.length === 9 ? parsed : [1,0,0, 0,1,0, 0,0,1];
      }
    }
    static identity() { return new Mat3(); }
    static fromTranslation(position) {
      const result = new Mat3();
      result.m[6] = position.x; result.m[7] = position.y;
      return result;
    }
    static fromScale(scale) {
      const result = new Mat3();
      if (typeof scale === 'number') {
        result.m[0] = scale; result.m[4] = scale;
      } else {
        result.m[0] = scale.x; result.m[4] = scale.y;
      }
      return result;
    }
    static fromRotation(angle) {
      const radians = angle * (Math.PI / 180);
      const cosine = Math.cos(radians), sine = Math.sin(radians);
      const result = new Mat3();
      result.m[0] = cosine; result.m[1] = sine;
      result.m[3] = -sine; result.m[4] = cosine;
      return result;
    }
    static fromBasis(right, up) {
      const result = new Mat3();
      result.m[0] = right.x; result.m[1] = right.y; result.m[2] = 0;
      result.m[3] = up.x; result.m[4] = up.y; result.m[5] = 0;
      result.m[6] = 0; result.m[7] = 0; result.m[8] = 1;
      return result;
    }
    static fromMat4(matrix) {
      const m = matrix.m;
      const result = new Mat3();
      result.m[0] = m[0]; result.m[1] = m[1]; result.m[2] = m[2];
      result.m[3] = m[4]; result.m[4] = m[5]; result.m[5] = m[6];
      result.m[6] = m[8]; result.m[7] = m[9]; result.m[8] = m[10];
      return result;
    }
    static compose(translation, rotation, scale) {
      return Mat3.fromTranslation(translation).rotate(rotation).scale(scale);
    }
    right() { return new Vec3(this.m[0], this.m[1], this.m[2]); }
    up() { return new Vec3(this.m[3], this.m[4], this.m[5]); }
    forward() { return new Vec3(this.m[6], this.m[7], this.m[8]); }
    translation(position) {
      if (position instanceof Vec2) {
        this.m[6] = position.x; this.m[7] = position.y;
        return this;
      }
      return new Vec2(this.m[6], this.m[7]);
    }
    // The heading accessor measures the stored basis with its own axis convention. It is
    // distinct from the rotation returned by decompose(), which factors signed X scale.
    angle() { return Math.atan2(this.m[0], -this.m[1]) * (180 / Math.PI); }
    add(other) {
      const result = new Mat3();
      for (let index = 0; index < 9; ++index) result.m[index] = this.m[index] + other.m[index];
      return result;
    }
    subtract(other) {
      const result = new Mat3();
      for (let index = 0; index < 9; ++index) result.m[index] = this.m[index] - other.m[index];
      return result;
    }
    multiply(other) {
      if (other instanceof Mat3) {
        const a = this.m, b = other.m;
        const result = new Mat3();
        for (let column = 0; column < 3; ++column) {
          for (let row = 0; row < 3; ++row) {
            let value = 0;
            for (let inner = 0; inner < 3; ++inner) {
              value += a[inner * 3 + row] * b[column * 3 + inner];
            }
            result.m[column * 3 + row] = value;
          }
        }
        return result;
      } else if (other instanceof Vec3) {
        const m = this.m;
        return new Vec3(m[0]*other.x + m[3]*other.y + m[6]*other.z,
                        m[1]*other.x + m[4]*other.y + m[7]*other.z,
                        m[2]*other.x + m[5]*other.y + m[8]*other.z);
      } else if (typeof other === 'number') {
        const result = new Mat3();
        for (let index = 0; index < 9; ++index) result.m[index] = this.m[index] * other;
        return result;
      }
    }
    translate(position) {
      const m = this.m;
      const result = new Mat3();
      result.m[0] = m[0]; result.m[1] = m[1]; result.m[2] = m[2];
      result.m[3] = m[3]; result.m[4] = m[4]; result.m[5] = m[5];
      result.m[6] = m[0]*position.x + m[3]*position.y + m[6];
      result.m[7] = m[1]*position.x + m[4]*position.y + m[7];
      result.m[8] = m[2]*position.x + m[5]*position.y + m[8];
      return result;
    }
    rotate(angle) {
      const radians = angle * (Math.PI / 180);
      const cosine = Math.cos(radians), sine = Math.sin(radians);
      const m = this.m;
      const result = new Mat3();
      result.m[0] = cosine*m[0] + sine*m[3];
      result.m[1] = cosine*m[1] + sine*m[4];
      result.m[2] = cosine*m[2] + sine*m[5];
      result.m[3] = -sine*m[0] + cosine*m[3];
      result.m[4] = -sine*m[1] + cosine*m[4];
      result.m[5] = -sine*m[2] + cosine*m[5];
      result.m[6] = m[6]; result.m[7] = m[7]; result.m[8] = m[8];
      return result;
    }
    scale(scale) {
      const m = this.m;
      const x = typeof scale === 'number' ? scale : scale.x;
      const y = typeof scale === 'number' ? scale : scale.y;
      const result = new Mat3();
      result.m[0] = m[0]*x; result.m[1] = m[1]*x; result.m[2] = m[2]*x;
      result.m[3] = m[3]*y; result.m[4] = m[4]*y; result.m[5] = m[5]*y;
      result.m[6] = m[6]; result.m[7] = m[7]; result.m[8] = m[8];
      return result;
    }
    transformPoint(point) {
      const m = this.m;
      return new Vec2(m[0]*point.x + m[3]*point.y + m[6],
                      m[1]*point.x + m[4]*point.y + m[7]);
    }
    transformDirection(direction) {
      const m = this.m;
      return new Vec2(m[0]*direction.x + m[3]*direction.y,
                      m[1]*direction.x + m[4]*direction.y);
    }
    transpose() {
      const m = this.m;
      const result = new Mat3();
      result.m[0] = m[0]; result.m[1] = m[3]; result.m[2] = m[6];
      result.m[3] = m[1]; result.m[4] = m[4]; result.m[5] = m[7];
      result.m[6] = m[2]; result.m[7] = m[5]; result.m[8] = m[8];
      return result;
    }
    determinant() {
      const m = this.m;
      return m[0] * (m[4]*m[8] - m[7]*m[5])
           - m[3] * (m[1]*m[8] - m[7]*m[2])
           + m[6] * (m[1]*m[5] - m[4]*m[2]);
    }
    inverse() {
      const m = this.m;
      const c00 = m[4]*m[8] - m[7]*m[5];
      const c01 = -(m[1]*m[8] - m[7]*m[2]);
      const c02 = m[1]*m[5] - m[4]*m[2];
      const c10 = -(m[3]*m[8] - m[6]*m[5]);
      const c11 = m[0]*m[8] - m[6]*m[2];
      const c12 = -(m[0]*m[5] - m[3]*m[2]);
      const c20 = m[3]*m[7] - m[6]*m[4];
      const c21 = -(m[0]*m[7] - m[6]*m[1]);
      const c22 = m[0]*m[4] - m[3]*m[1];
      const determinant = m[0]*c00 + m[3]*c01 + m[6]*c02;
      // Preserve ordinary JavaScript division for singular matrices; callers observe its
      // non-finite components instead of receiving a substituted transform or exception.
      const reciprocal = 1 / determinant;
      const result = new Mat3();
      result.m[0] = c00*reciprocal; result.m[1] = c01*reciprocal; result.m[2] = c02*reciprocal;
      result.m[3] = c10*reciprocal; result.m[4] = c11*reciprocal; result.m[5] = c12*reciprocal;
      result.m[6] = c20*reciprocal; result.m[7] = c21*reciprocal; result.m[8] = c22*reciprocal;
      return result;
    }
    decompose() {
      const m = this.m;
      const translation = new Vec2(m[6], m[7]);
      const x = Math.sqrt(m[0]*m[0] + m[1]*m[1]);
      const y = Math.sqrt(m[3]*m[3] + m[4]*m[4]);
      const determinant = m[0]*m[4] - m[3]*m[1];
      // A reflection is carried by signed X scale. This keeps the returned rotation and
      // scale usable together without treating a reflected basis as an extra half-turn.
      const signedX = determinant < 0 ? -x : x;
      const rotation = signedX !== 0
        ? Math.atan2(m[1]/signedX, m[0]/signedX) * (180 / Math.PI) : 0;
      return { translation, rotation, scale: new Vec2(signedX, y) };
    }
    copy() { return new Mat3(this); }
    equals(other) {
      if (!(other instanceof Mat3)) return false;
      for (let index = 0; index < 9; ++index) {
        if (Math.abs(this.m[index] - other.m[index]) >= __vectorEpsilon) return false;
      }
      return true;
    }
    toString() {
      return this.m[0]+' '+this.m[1]+' '+this.m[2]+' '+
             this.m[3]+' '+this.m[4]+' '+this.m[5]+' '+
             this.m[6]+' '+this.m[7]+' '+this.m[8];
    }
    toConfigString() { return this.toString(); }
  });
)JS";

} // namespace wallpaper
