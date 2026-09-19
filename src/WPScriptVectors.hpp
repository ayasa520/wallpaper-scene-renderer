#pragma once

#include <string_view>

namespace wallpaper {

// Both callback wrappers share these vector declarations. Constructors remain unique within
// a JavaScript realm, so values created by different layer callbacks retain their identity.
// Scalar parsing and arithmetic helpers come from the wrapper's existing numeric environment.
inline constexpr std::string_view kSceneScriptVectorPrelude = R"JS(
  const Vec2 = (typeof globalThis.Vec2 === 'function')
    ? globalThis.Vec2
    : (globalThis.Vec2 = class Vec2 {
        constructor(a = 0, b = 0) {
          const values = arguments.length === 0 ? [0, 0]
            : arguments.length === 1 ? __vecValues(a, 2)
            : [__toNumber(a, 0), __toNumber(b, 0)];
          this.x = values[0];
          this.y = values[1];
        }
        equals(other) { const v = __vecValues(other, 2); return Math.abs(this.x - v[0]) < 1e-6 && Math.abs(this.y - v[1]) < 1e-6; }
        length() { return Math.hypot(this.x, this.y); }
        lengthSqr() { return this.x * this.x + this.y * this.y; }
        normalize() { const len = this.length(); return len === 0 ? new Vec2() : new Vec2(this.x / len, this.y / len); }
        copy() { return new Vec2(this.x, this.y); }
        add(value) { return __binaryVec(this, value, (a, b) => a + b, Vec2, ['x', 'y']); }
        subtract(value) { return __binaryVec(this, value, (a, b) => a - b, Vec2, ['x', 'y']); }
        multiply(value) { return __binaryVec(this, value, (a, b) => a * b, Vec2, ['x', 'y']); }
        divide(value) { return __binaryVec(this, value, (a, b) => a / b, Vec2, ['x', 'y']); }
        dot(value) { const rhs = __vecValues(value, 2); return __dot([this.x, this.y], rhs); }
        reflect(normal) { const n = new Vec2(normal).normalize(); return this.subtract(n.multiply(2 * this.dot(n))); }
        mix(other, amount) { const rhs = __vecValues(other, 2); return new Vec2(__mixScalar(this.x, rhs[0], amount), __mixScalar(this.y, rhs[1], amount)); }
        min(value) { return __binaryVec(this, value, (a, b) => Math.min(a, b), Vec2, ['x', 'y']); }
        max(value) { return __binaryVec(this, value, (a, b) => Math.max(a, b), Vec2, ['x', 'y']); }
        perpendicular() { return new Vec2(-this.y, this.x); }
        abs() { return new Vec2(Math.abs(this.x), Math.abs(this.y)); }
        sign() { return new Vec2(Math.sign(this.x), Math.sign(this.y)); }
        round() { return new Vec2(Math.round(this.x), Math.round(this.y)); }
        floor() { return new Vec2(Math.floor(this.x), Math.floor(this.y)); }
        ceil() { return new Vec2(Math.ceil(this.x), Math.ceil(this.y)); }
        toString() { return `${this.x} ${this.y}`; }
        toConfigString() { return this.toString(); }
      });
  const Vec3 = (typeof globalThis.Vec3 === 'function')
    ? globalThis.Vec3
    : (globalThis.Vec3 = class Vec3 {
        constructor(a = 0, b = 0, c = 0) {
          const values = arguments.length === 0 ? [0, 0, 0]
            : arguments.length === 1 ? __vecValues(a, 3)
            : arguments.length === 2 ? [__toNumber(a, 0), __toNumber(b, 0), 0]
            : [__toNumber(a, 0), __toNumber(b, 0), __toNumber(c, 0)];
          this.x = values[0];
          this.y = values[1];
          this.z = values[2];
        }
        equals(other) { const v = __vecValues(other, 3); return Math.abs(this.x - v[0]) < 1e-6 && Math.abs(this.y - v[1]) < 1e-6 && Math.abs(this.z - v[2]) < 1e-6; }
        length() { return Math.hypot(this.x, this.y, this.z); }
        lengthSqr() { return this.x * this.x + this.y * this.y + this.z * this.z; }
        normalize() { const len = this.length(); return len === 0 ? new Vec3() : new Vec3(this.x / len, this.y / len, this.z / len); }
        copy() { return new Vec3(this.x, this.y, this.z); }
        add(value) { return __binaryVec(this, value, (a, b) => a + b, Vec3, ['x', 'y', 'z']); }
        subtract(value) { return __binaryVec(this, value, (a, b) => a - b, Vec3, ['x', 'y', 'z']); }
        multiply(value) { return __binaryVec(this, value, (a, b) => a * b, Vec3, ['x', 'y', 'z']); }
        divide(value) { return __binaryVec(this, value, (a, b) => a / b, Vec3, ['x', 'y', 'z']); }
        dot(value) { const rhs = __vecValues(value, 3); return __dot([this.x, this.y, this.z], rhs); }
        reflect(normal) { const n = new Vec3(normal).normalize(); return this.subtract(n.multiply(2 * this.dot(n))); }
        mix(other, amount) { const rhs = __vecValues(other, 3); return new Vec3(__mixScalar(this.x, rhs[0], amount), __mixScalar(this.y, rhs[1], amount), __mixScalar(this.z, rhs[2], amount)); }
        min(value) { return __binaryVec(this, value, (a, b) => Math.min(a, b), Vec3, ['x', 'y', 'z']); }
        max(value) { return __binaryVec(this, value, (a, b) => Math.max(a, b), Vec3, ['x', 'y', 'z']); }
        cross(value) { const rhs = __vecValues(value, 3); return new Vec3(this.y * rhs[2] - this.z * rhs[1], this.z * rhs[0] - this.x * rhs[2], this.x * rhs[1] - this.y * rhs[0]); }
        abs() { return new Vec3(Math.abs(this.x), Math.abs(this.y), Math.abs(this.z)); }
        sign() { return new Vec3(Math.sign(this.x), Math.sign(this.y), Math.sign(this.z)); }
        round() { return new Vec3(Math.round(this.x), Math.round(this.y), Math.round(this.z)); }
        floor() { return new Vec3(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z)); }
        ceil() { return new Vec3(Math.ceil(this.x), Math.ceil(this.y), Math.ceil(this.z)); }
        toString() { return `${this.x} ${this.y} ${this.z}`; }
        toConfigString() { return this.toString(); }
      });
  const Vec4 = (globalThis.Vec4 ??= class Vec4 {
    constructor(x, y, z, w) {
      let components;
      if (typeof x === 'string') {
        const words = x.split(' ');
        components = [parseFloat(words[0]), parseFloat(words[1]),
                      parseFloat(words[2]), parseFloat(words[3])];
      } else if (x instanceof Vec4) {
        components = [x.x, x.y, x.z, x.w];
      } else if (x instanceof Vec3) {
        components = [x.x, x.y, x.z, 0];
      } else if (x instanceof Vec2) {
        components = [x.x, x.y, 0, 0];
      } else if (x === undefined) {
        components = [0, 0, 0, 0];
      } else {
        // Scalar construction fills all components. With two numeric arguments the unused
        // components are zero; a third numeric argument supplies both z and the omitted w.
        // Preserve supplied component values instead of applying a finite-number coercion.
        const hasY = typeof y === 'number';
        const hasZ = typeof z === 'number';
        components = [x, hasY ? y : x, hasZ ? z : (hasY ? 0 : x),
                      typeof w === 'number' ? w : (hasZ ? z : (hasY ? 0 : x))];
      }
      [this.x, this.y, this.z, this.w] = components;
    }
    toString() { return this.x + ' ' + this.y + ' ' + this.z + ' ' + this.w; }
    toConfigString() { return this.toString(); }
  });
)JS";

} // namespace wallpaper
