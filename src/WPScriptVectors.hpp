#pragma once

#include <string_view>

namespace wallpaper {

// Both callback wrappers share these vector declarations. Constructors remain unique within
// a JavaScript realm, so values created by different layer callbacks retain their identity.
// Vector components retain JavaScript values, including non-finite numbers. Host-side numeric
// parsing is separate: applying it inside these operations would change their arithmetic.
inline constexpr std::string_view kSceneScriptVectorPrelude = R"JS(
  const __vectorEpsilon = 0.00001;
  const Vec2 = (typeof globalThis.Vec2 === 'function')
    ? globalThis.Vec2
    : (globalThis.Vec2 = class Vec2 {
        constructor(x, y) {
          if (typeof x === 'string') {
            const words = x.split(' ');
            this.x = parseFloat(words[0]);
            this.y = parseFloat(words[1]);
          } else if (x instanceof Vec3 || x instanceof Vec2) {
            this.x = x.x;
            this.y = x.y;
          } else if (x !== undefined) {
            this.x = x;
            this.y = typeof y === 'number' ? y : x;
          } else {
            this.x = 0;
            this.y = 0;
          }
        }
        equals(other) { return other instanceof Vec2 && Math.abs(this.x - other.x) < __vectorEpsilon && Math.abs(this.y - other.y) < __vectorEpsilon; }
        length() { return Math.sqrt(this.x * this.x + this.y * this.y); }
        lengthSqr() { return this.x * this.x + this.y * this.y; }
        normalize() { return this.divide(this.length()); }
        copy() { return new Vec2(this.x, this.y); }
        add(value) { return typeof value === 'number' ? new Vec2(this.x + value, this.y + value) : new Vec2(this.x + value.x, this.y + value.y); }
        subtract(value) { return typeof value === 'number' ? new Vec2(this.x - value, this.y - value) : new Vec2(this.x - value.x, this.y - value.y); }
        multiply(value) { return typeof value === 'number' ? new Vec2(this.x * value, this.y * value) : new Vec2(this.x * value.x, this.y * value.y); }
        divide(value) { return typeof value === 'number' ? new Vec2(this.x / value, this.y / value) : new Vec2(this.x / value.x, this.y / value.y); }
        dot(value) { return this.x * value.x + this.y * value.y; }
        reflect(normal) { return this.subtract(normal.multiply(2 * this.dot(normal))); }
        mix(other, amount) { return new Vec2(this.x + (other.x - this.x) * (typeof amount === 'number' ? amount : amount.x), this.y + (other.y - this.y) * (typeof amount === 'number' ? amount : amount.y)); }
        min(value) { return typeof value === 'number' ? new Vec2(Math.min(this.x, value), Math.min(this.y, value)) : new Vec2(Math.min(this.x, value.x), Math.min(this.y, value.y)); }
        max(value) { return typeof value === 'number' ? new Vec2(Math.max(this.x, value), Math.max(this.y, value)) : new Vec2(Math.max(this.x, value.x), Math.max(this.y, value.y)); }
        perpendicular() { return new Vec2(this.y, -this.x); }
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
        constructor(x, y, z) {
          if (typeof x === 'string') {
            const words = x.split(' ');
            this.x = parseFloat(words[0]);
            this.y = parseFloat(words[1]);
            this.z = parseFloat(words[2]);
          } else if (x instanceof Vec3 || x instanceof Vec2) {
            this.x = x.x;
            this.y = x.y;
            this.z = x instanceof Vec3 ? x.z : 0;
          } else if (x !== undefined) {
            this.x = x;
            this.y = typeof y === 'number' ? y : x;
            this.z = typeof z === 'number' ? z : (typeof y === 'number' ? 0 : x);
          } else {
            this.x = 0;
            this.y = 0;
            this.z = 0;
          }
        }
        equals(other) { return other instanceof Vec3 && Math.abs(this.x - other.x) < __vectorEpsilon && Math.abs(this.y - other.y) < __vectorEpsilon && Math.abs(this.z - other.z) < __vectorEpsilon; }
        length() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z); }
        lengthSqr() { return this.x * this.x + this.y * this.y + this.z * this.z; }
        normalize() { return this.divide(this.length()); }
        copy() { return new Vec3(this.x, this.y, this.z); }
        // A Vec2 operand changes only the first two components for these four operations.
        // Read each component directly so non-finite arithmetic is preserved in the result.
        add(value) { return typeof value === 'number' ? new Vec3(this.x + value, this.y + value, this.z + value) : new Vec3(this.x + value.x, this.y + value.y, value instanceof Vec2 ? this.z : this.z + value.z); }
        subtract(value) { return typeof value === 'number' ? new Vec3(this.x - value, this.y - value, this.z - value) : new Vec3(this.x - value.x, this.y - value.y, value instanceof Vec2 ? this.z : this.z - value.z); }
        multiply(value) { return typeof value === 'number' ? new Vec3(this.x * value, this.y * value, this.z * value) : new Vec3(this.x * value.x, this.y * value.y, value instanceof Vec2 ? this.z : this.z * value.z); }
        divide(value) { return typeof value === 'number' ? new Vec3(this.x / value, this.y / value, this.z / value) : new Vec3(this.x / value.x, this.y / value.y, value instanceof Vec2 ? this.z : this.z / value.z); }
        dot(value) { return this.x * value.x + this.y * value.y + this.z * value.z; }
        reflect(normal) { return this.subtract(normal.multiply(2 * this.dot(normal))); }
        mix(other, amount) { return new Vec3(this.x + (other.x - this.x) * (typeof amount === 'number' ? amount : amount.x), this.y + (other.y - this.y) * (typeof amount === 'number' ? amount : amount.y), this.z + (other.z - this.z) * (typeof amount === 'number' ? amount : amount.z)); }
        min(value) { return typeof value === 'number' ? new Vec3(Math.min(this.x, value), Math.min(this.y, value), Math.min(this.z, value)) : new Vec3(Math.min(this.x, value.x), Math.min(this.y, value.y), Math.min(this.z, value.z)); }
        max(value) { return typeof value === 'number' ? new Vec3(Math.max(this.x, value), Math.max(this.y, value), Math.max(this.z, value)) : new Vec3(Math.max(this.x, value.x), Math.max(this.y, value.y), Math.max(this.z, value.z)); }
        cross(value) { return new Vec3(this.y * value.z - this.z * value.y, this.z * value.x - this.x * value.z, this.x * value.y - this.y * value.x); }
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
    // Arithmetic owns a fresh result and leaves both inputs untouched. Read component operands
    // directly so JavaScript operators retain their ordinary conversion and evaluation order.
    add(value) {
      if (typeof value === 'number') {
        return new Vec4(this.x + value, this.y + value, this.z + value, this.w + value);
      }
      return new Vec4(this.x + value.x, this.y + value.y, this.z + value.z, this.w + value.w);
    }
    subtract(value) {
      if (typeof value === 'number') {
        return new Vec4(this.x - value, this.y - value, this.z - value, this.w - value);
      }
      return new Vec4(this.x - value.x, this.y - value.y, this.z - value.z, this.w - value.w);
    }
    multiply(value) {
      if (typeof value === 'number') {
        return new Vec4(this.x * value, this.y * value, this.z * value, this.w * value);
      }
      return new Vec4(this.x * value.x, this.y * value.y, this.z * value.z, this.w * value.w);
    }
    divide(value) {
      if (typeof value === 'number') {
        return new Vec4(this.x / value, this.y / value, this.z / value, this.w / value);
      }
      return new Vec4(this.x / value.x, this.y / value.y, this.z / value.z, this.w / value.w);
    }
    length() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w); }
    normalize() { return this.divide(this.length()); }
    toString() { return this.x + ' ' + this.y + ' ' + this.z + ' ' + this.w; }
    toConfigString() { return this.toString(); }
  });
)JS";

} // namespace wallpaper
