#pragma once

#include <string_view>

namespace wallpaper {

// Both callback wrappers share these vector declarations. Constructors remain unique within
// a JavaScript realm, so values created by different layer callbacks retain their identity.
// Vector components retain JavaScript values, including non-finite numbers. Host-side numeric
// parsing is separate: applying it inside these operations would change their arithmetic.
inline constexpr std::string_view kSceneScriptVectorPrelude = R"JS(
  const __vectorEpsilon = 0.00001;
  const __vectorDegreesToRadians = Math.PI / 180;
  const __vectorRadiansToDegrees = 180 / Math.PI;
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
        distance(value) {
          const dx = this.x - value.x, dy = this.y - value.y;
          return Math.sqrt(dx * dx + dy * dy);
        }
        distanceSqr(value) {
          const dx = this.x - value.x, dy = this.y - value.y;
          return dx * dx + dy * dy;
        }
        normalize() { return this.divide(this.length()); }
        copy() { return new Vec2(this.x, this.y); }
        isFinite() { return Number.isFinite(this.x) && Number.isFinite(this.y); }
        negate() { return new Vec2(-this.x, -this.y); }
        add(value) { return typeof value === 'number' ? new Vec2(this.x + value, this.y + value) : new Vec2(this.x + value.x, this.y + value.y); }
        subtract(value) { return typeof value === 'number' ? new Vec2(this.x - value, this.y - value) : new Vec2(this.x - value.x, this.y - value.y); }
        multiply(value) { return typeof value === 'number' ? new Vec2(this.x * value, this.y * value) : new Vec2(this.x * value.x, this.y * value.y); }
        divide(value) { return typeof value === 'number' ? new Vec2(this.x / value, this.y / value) : new Vec2(this.x / value.x, this.y / value.y); }
        dot(value) { return this.x * value.x + this.y * value.y; }
        reflect(normal) { return this.subtract(normal.multiply(2 * this.dot(normal))); }
        // Projection onto a zero vector has an explicit zero result. Every vector-valued
        // operation owns its result, including this case, and never mutates either input.
        project(value) {
          const squared = value.lengthSqr();
          if (squared === 0) return new Vec2(0, 0);
          return value.multiply(this.dot(value) / squared);
        }
        // Two-dimensional angles retain their sign and use degrees at the script boundary.
        angle() { return Math.atan2(this.y, this.x) * __vectorRadiansToDegrees; }
        angleBetween(value) {
          return Math.atan2(this.x * value.y - this.y * value.x,
                            this.x * value.x + this.y * value.y) * __vectorRadiansToDegrees;
        }
        rotate(angle) {
          const radians = angle * __vectorDegreesToRadians;
          const cosine = Math.cos(radians), sine = Math.sin(radians);
          return new Vec2(cosine * this.x - sine * this.y, sine * this.x + cosine * this.y);
        }
        mix(other, amount) { return new Vec2(this.x + (other.x - this.x) * (typeof amount === 'number' ? amount : amount.x), this.y + (other.y - this.y) * (typeof amount === 'number' ? amount : amount.y)); }
        min(value) { return typeof value === 'number' ? new Vec2(Math.min(this.x, value), Math.min(this.y, value)) : new Vec2(Math.min(this.x, value.x), Math.min(this.y, value.y)); }
        max(value) { return typeof value === 'number' ? new Vec2(Math.max(this.x, value), Math.max(this.y, value)) : new Vec2(Math.max(this.x, value.x), Math.max(this.y, value.y)); }
        // Each bound independently accepts a scalar or matching vector. Preserve the
        // component arithmetic rather than coercing non-finite values at the host boundary.
        clamp(minimum, maximum) {
          const minX = typeof minimum === 'number' ? minimum : minimum.x;
          const minY = typeof minimum === 'number' ? minimum : minimum.y;
          const maxX = typeof maximum === 'number' ? maximum : maximum.x;
          const maxY = typeof maximum === 'number' ? maximum : maximum.y;
          return new Vec2(Math.max(minX, Math.min(maxX, this.x)),
                          Math.max(minY, Math.min(maxY, this.y)));
        }
        perpendicular() { return new Vec2(this.y, -this.x); }
        abs() { return new Vec2(Math.abs(this.x), Math.abs(this.y)); }
        sign() { return new Vec2(Math.sign(this.x), Math.sign(this.y)); }
        round() { return new Vec2(Math.round(this.x), Math.round(this.y)); }
        floor() { return new Vec2(Math.floor(this.x), Math.floor(this.y)); }
        ceil() { return new Vec2(Math.ceil(this.x), Math.ceil(this.y)); }
        fract() { return new Vec2(this.x - Math.floor(this.x), this.y - Math.floor(this.y)); }
        // Floor-based modulo preserves the divisor's sign convention for negative inputs.
        mod(value) {
          if (typeof value === 'number') {
            return new Vec2(this.x - value * Math.floor(this.x / value),
                            this.y - value * Math.floor(this.y / value));
          }
          return new Vec2(this.x - value.x * Math.floor(this.x / value.x),
                          this.y - value.y * Math.floor(this.y / value.y));
        }
        step(edge) {
          const x = typeof edge === 'number' ? edge : edge.x;
          const y = typeof edge === 'number' ? edge : edge.y;
          return new Vec2(this.x < x ? 0 : 1, this.y < y ? 0 : 1);
        }
        smoothStep(minimum, maximum) {
          const minX = typeof minimum === 'number' ? minimum : minimum.x;
          const minY = typeof minimum === 'number' ? minimum : minimum.y;
          const maxX = typeof maximum === 'number' ? maximum : maximum.x;
          const maxY = typeof maximum === 'number' ? maximum : maximum.y;
          const x = Math.max(0, Math.min(1, (this.x - minX) / (maxX - minX)));
          const y = Math.max(0, Math.min(1, (this.y - minY) / (maxY - minY)));
          return new Vec2(x * x * (3 - 2 * x), y * y * (3 - 2 * y));
        }
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
        // Spherical tuples use radius, polar angle from +Y, and azimuth around Y. Angles
        // enter and leave this API in degrees; the trigonometric functions use radians.
        static fromSpherical(radius, theta, phi) {
          const polar = theta * __vectorDegreesToRadians;
          const azimuth = phi * __vectorDegreesToRadians;
          const sine = Math.sin(polar);
          return new Vec3(radius * sine * Math.cos(azimuth), radius * Math.cos(polar),
                          radius * sine * Math.sin(azimuth));
        }
        equals(other) { return other instanceof Vec3 && Math.abs(this.x - other.x) < __vectorEpsilon && Math.abs(this.y - other.y) < __vectorEpsilon && Math.abs(this.z - other.z) < __vectorEpsilon; }
        length() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z); }
        lengthSqr() { return this.x * this.x + this.y * this.y + this.z * this.z; }
        distance(value) {
          const dx = this.x - value.x, dy = this.y - value.y, dz = this.z - value.z;
          return Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        distanceSqr(value) {
          const dx = this.x - value.x, dy = this.y - value.y, dz = this.z - value.z;
          return dx * dx + dy * dy + dz * dz;
        }
        normalize() { return this.divide(this.length()); }
        copy() { return new Vec3(this.x, this.y, this.z); }
        isFinite() { return Number.isFinite(this.x) && Number.isFinite(this.y) && Number.isFinite(this.z); }
        negate() { return new Vec3(-this.x, -this.y, -this.z); }
        // A Vec2 operand changes only the first two components for these four operations.
        // Read each component directly so non-finite arithmetic is preserved in the result.
        add(value) { return typeof value === 'number' ? new Vec3(this.x + value, this.y + value, this.z + value) : new Vec3(this.x + value.x, this.y + value.y, value instanceof Vec2 ? this.z : this.z + value.z); }
        subtract(value) { return typeof value === 'number' ? new Vec3(this.x - value, this.y - value, this.z - value) : new Vec3(this.x - value.x, this.y - value.y, value instanceof Vec2 ? this.z : this.z - value.z); }
        multiply(value) { return typeof value === 'number' ? new Vec3(this.x * value, this.y * value, this.z * value) : new Vec3(this.x * value.x, this.y * value.y, value instanceof Vec2 ? this.z : this.z * value.z); }
        divide(value) { return typeof value === 'number' ? new Vec3(this.x / value, this.y / value, this.z / value) : new Vec3(this.x / value.x, this.y / value.y, value instanceof Vec2 ? this.z : this.z / value.z); }
        dot(value) { return this.x * value.x + this.y * value.y + this.z * value.z; }
        reflect(normal) { return this.subtract(normal.multiply(2 * this.dot(normal))); }
        // Total internal reflection and projection onto a zero direction return fresh zero
        // vectors. Ordinary refraction consumes the supplied incident and normal vectors.
        refract(normal, eta) {
          const normalDotIncident = normal.dot(this);
          const discriminant = 1 - eta * eta * (1 - normalDotIncident * normalDotIncident);
          if (discriminant < 0) return new Vec3(0, 0, 0);
          return this.multiply(eta).subtract(normal.multiply(eta * normalDotIncident + Math.sqrt(discriminant)));
        }
        project(value) {
          const squared = value.lengthSqr();
          if (squared === 0) return new Vec3(0, 0, 0);
          return value.multiply(this.dot(value) / squared);
        }
        // A three-dimensional angle is unsigned. Clamp the normalized dot product for
        // acos, while the explicitly defined zero-length case returns zero degrees.
        angleBetween(value) {
          const denominator = Math.sqrt(this.lengthSqr() * value.lengthSqr());
          if (denominator === 0) return 0;
          return Math.acos(Math.max(-1, Math.min(1, this.dot(value) / denominator))) * __vectorRadiansToDegrees;
        }
        toSpherical() {
          const radius = this.length();
          if (radius === 0) return new Vec3(0, 0, 0);
          return new Vec3(radius, Math.acos(this.y / radius) * __vectorRadiansToDegrees,
                          Math.atan2(this.z, this.x) * __vectorRadiansToDegrees);
        }
        mix(other, amount) { return new Vec3(this.x + (other.x - this.x) * (typeof amount === 'number' ? amount : amount.x), this.y + (other.y - this.y) * (typeof amount === 'number' ? amount : amount.y), this.z + (other.z - this.z) * (typeof amount === 'number' ? amount : amount.z)); }
        min(value) { return typeof value === 'number' ? new Vec3(Math.min(this.x, value), Math.min(this.y, value), Math.min(this.z, value)) : new Vec3(Math.min(this.x, value.x), Math.min(this.y, value.y), Math.min(this.z, value.z)); }
        max(value) { return typeof value === 'number' ? new Vec3(Math.max(this.x, value), Math.max(this.y, value), Math.max(this.z, value)) : new Vec3(Math.max(this.x, value.x), Math.max(this.y, value.y), Math.max(this.z, value.z)); }
        // Scalar/vector bounds are independent, matching the component-wise Vec2 contract.
        clamp(minimum, maximum) {
          const minX = typeof minimum === 'number' ? minimum : minimum.x;
          const minY = typeof minimum === 'number' ? minimum : minimum.y;
          const minZ = typeof minimum === 'number' ? minimum : minimum.z;
          const maxX = typeof maximum === 'number' ? maximum : maximum.x;
          const maxY = typeof maximum === 'number' ? maximum : maximum.y;
          const maxZ = typeof maximum === 'number' ? maximum : maximum.z;
          return new Vec3(Math.max(minX, Math.min(maxX, this.x)),
                          Math.max(minY, Math.min(maxY, this.y)),
                          Math.max(minZ, Math.min(maxZ, this.z)));
        }
        cross(value) { return new Vec3(this.y * value.z - this.z * value.y, this.z * value.x - this.x * value.z, this.x * value.y - this.y * value.x); }
        abs() { return new Vec3(Math.abs(this.x), Math.abs(this.y), Math.abs(this.z)); }
        sign() { return new Vec3(Math.sign(this.x), Math.sign(this.y), Math.sign(this.z)); }
        round() { return new Vec3(Math.round(this.x), Math.round(this.y), Math.round(this.z)); }
        floor() { return new Vec3(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z)); }
        ceil() { return new Vec3(Math.ceil(this.x), Math.ceil(this.y), Math.ceil(this.z)); }
        fract() {
          return new Vec3(this.x - Math.floor(this.x), this.y - Math.floor(this.y),
                          this.z - Math.floor(this.z));
        }
        mod(value) {
          if (typeof value === 'number') {
            return new Vec3(this.x - value * Math.floor(this.x / value),
                            this.y - value * Math.floor(this.y / value),
                            this.z - value * Math.floor(this.z / value));
          }
          return new Vec3(this.x - value.x * Math.floor(this.x / value.x),
                          this.y - value.y * Math.floor(this.y / value.y),
                          this.z - value.z * Math.floor(this.z / value.z));
        }
        step(edge) {
          const x = typeof edge === 'number' ? edge : edge.x;
          const y = typeof edge === 'number' ? edge : edge.y;
          const z = typeof edge === 'number' ? edge : edge.z;
          return new Vec3(this.x < x ? 0 : 1, this.y < y ? 0 : 1, this.z < z ? 0 : 1);
        }
        smoothStep(minimum, maximum) {
          const minX = typeof minimum === 'number' ? minimum : minimum.x;
          const minY = typeof minimum === 'number' ? minimum : minimum.y;
          const minZ = typeof minimum === 'number' ? minimum : minimum.z;
          const maxX = typeof maximum === 'number' ? maximum : maximum.x;
          const maxY = typeof maximum === 'number' ? maximum : maximum.y;
          const maxZ = typeof maximum === 'number' ? maximum : maximum.z;
          const x = Math.max(0, Math.min(1, (this.x - minX) / (maxX - minX)));
          const y = Math.max(0, Math.min(1, (this.y - minY) / (maxY - minY)));
          const z = Math.max(0, Math.min(1, (this.z - minZ) / (maxZ - minZ)));
          return new Vec3(x * x * (3 - 2 * x), y * y * (3 - 2 * y), z * z * (3 - 2 * z));
        }
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
    dot(value) { return this.x * value.x + this.y * value.y + this.z * value.z + this.w * value.w; }
    length() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w); }
    lengthSqr() { return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w; }
    distance(value) {
      const dx = this.x - value.x, dy = this.y - value.y;
      const dz = this.z - value.z, dw = this.w - value.w;
      return Math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
    }
    distanceSqr(value) {
      const dx = this.x - value.x, dy = this.y - value.y;
      const dz = this.z - value.z, dw = this.w - value.w;
      return dx * dx + dy * dy + dz * dz + dw * dw;
    }
    normalize() { return this.divide(this.length()); }
    copy() { return new Vec4(this.x, this.y, this.z, this.w); }
    equals(value) {
      return value instanceof Vec4 && Math.abs(this.x - value.x) < __vectorEpsilon &&
        Math.abs(this.y - value.y) < __vectorEpsilon && Math.abs(this.z - value.z) < __vectorEpsilon &&
        Math.abs(this.w - value.w) < __vectorEpsilon;
    }
    isFinite() {
      return Number.isFinite(this.x) && Number.isFinite(this.y) &&
        Number.isFinite(this.z) && Number.isFinite(this.w);
    }
    negate() { return new Vec4(-this.x, -this.y, -this.z, -this.w); }
    reflect(normal) { return this.subtract(normal.multiply(2 * this.dot(normal))); }
    project(value) {
      // Projection onto a zero vector is defined as a fresh zero result. Nonzero
      // directions need not be normalized, and neither input is mutated by projection.
      const squared = value.lengthSqr();
      if (squared === 0) return new Vec4(0, 0, 0, 0);
      return value.multiply(this.dot(value) / squared);
    }
    // Interpolation owns its result, leaving both endpoints and the weight vector available
    // for later material updates. Evaluate each component in JavaScript number space; the
    // host applies its numeric storage conversion only when the result reaches a property.
    mix(other, amount) {
      if (typeof amount === 'number') {
        return new Vec4(this.x + (other.x - this.x) * amount,
                        this.y + (other.y - this.y) * amount,
                        this.z + (other.z - this.z) * amount,
                        this.w + (other.w - this.w) * amount);
      }
      return new Vec4(this.x + (other.x - this.x) * amount.x,
                      this.y + (other.y - this.y) * amount.y,
                      this.z + (other.z - this.z) * amount.z,
                      this.w + (other.w - this.w) * amount.w);
    }
    min(value) {
      if (typeof value === 'number') {
        return new Vec4(Math.min(this.x, value), Math.min(this.y, value),
                        Math.min(this.z, value), Math.min(this.w, value));
      }
      return new Vec4(Math.min(this.x, value.x), Math.min(this.y, value.y),
                      Math.min(this.z, value.z), Math.min(this.w, value.w));
    }
    max(value) {
      if (typeof value === 'number') {
        return new Vec4(Math.max(this.x, value), Math.max(this.y, value),
                        Math.max(this.z, value), Math.max(this.w, value));
      }
      return new Vec4(Math.max(this.x, value.x), Math.max(this.y, value.y),
                      Math.max(this.z, value.z), Math.max(this.w, value.w));
    }
    clamp(minimum, maximum) {
      // Each bound independently accepts a scalar or a component vector. Read both
      // complete bounds before constructing the result; no temporary vector is needed.
      const minX = typeof minimum === 'number' ? minimum : minimum.x;
      const minY = typeof minimum === 'number' ? minimum : minimum.y;
      const minZ = typeof minimum === 'number' ? minimum : minimum.z;
      const minW = typeof minimum === 'number' ? minimum : minimum.w;
      const maxX = typeof maximum === 'number' ? maximum : maximum.x;
      const maxY = typeof maximum === 'number' ? maximum : maximum.y;
      const maxZ = typeof maximum === 'number' ? maximum : maximum.z;
      const maxW = typeof maximum === 'number' ? maximum : maximum.w;
      return new Vec4(Math.max(minX, Math.min(maxX, this.x)),
                      Math.max(minY, Math.min(maxY, this.y)),
                      Math.max(minZ, Math.min(maxZ, this.z)),
                      Math.max(minW, Math.min(maxW, this.w)));
    }
    abs() { return new Vec4(Math.abs(this.x), Math.abs(this.y), Math.abs(this.z), Math.abs(this.w)); }
    sign() { return new Vec4(Math.sign(this.x), Math.sign(this.y), Math.sign(this.z), Math.sign(this.w)); }
    round() { return new Vec4(Math.round(this.x), Math.round(this.y), Math.round(this.z), Math.round(this.w)); }
    floor() { return new Vec4(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z), Math.floor(this.w)); }
    ceil() { return new Vec4(Math.ceil(this.x), Math.ceil(this.y), Math.ceil(this.z), Math.ceil(this.w)); }
    fract() {
      return new Vec4(this.x - Math.floor(this.x), this.y - Math.floor(this.y),
                      this.z - Math.floor(this.z), this.w - Math.floor(this.w));
    }
    mod(value) {
      // Use floor-based modulo so negative components follow shader arithmetic.
      // JavaScript's remainder operator gives a different result for signed operands.
      if (typeof value === 'number') {
        return new Vec4(this.x - value * Math.floor(this.x / value),
                        this.y - value * Math.floor(this.y / value),
                        this.z - value * Math.floor(this.z / value),
                        this.w - value * Math.floor(this.w / value));
      }
      return new Vec4(this.x - value.x * Math.floor(this.x / value.x),
                      this.y - value.y * Math.floor(this.y / value.y),
                      this.z - value.z * Math.floor(this.z / value.z),
                      this.w - value.w * Math.floor(this.w / value.w));
    }
    step(edge) {
      const x = typeof edge === 'number' ? edge : edge.x;
      const y = typeof edge === 'number' ? edge : edge.y;
      const z = typeof edge === 'number' ? edge : edge.z;
      const w = typeof edge === 'number' ? edge : edge.w;
      return new Vec4(this.x < x ? 0 : 1, this.y < y ? 0 : 1,
                      this.z < z ? 0 : 1, this.w < w ? 0 : 1);
    }
    smoothStep(minimum, maximum) {
      // Normalize and clamp each component against its independently selected bounds
      // before applying the cubic Hermite curve. Keep all arithmetic in number space;
      // material assignment is the later boundary that converts values for GPU storage.
      const minX = typeof minimum === 'number' ? minimum : minimum.x;
      const minY = typeof minimum === 'number' ? minimum : minimum.y;
      const minZ = typeof minimum === 'number' ? minimum : minimum.z;
      const minW = typeof minimum === 'number' ? minimum : minimum.w;
      const maxX = typeof maximum === 'number' ? maximum : maximum.x;
      const maxY = typeof maximum === 'number' ? maximum : maximum.y;
      const maxZ = typeof maximum === 'number' ? maximum : maximum.z;
      const maxW = typeof maximum === 'number' ? maximum : maximum.w;
      const x = Math.max(0, Math.min(1, (this.x - minX) / (maxX - minX)));
      const y = Math.max(0, Math.min(1, (this.y - minY) / (maxY - minY)));
      const z = Math.max(0, Math.min(1, (this.z - minZ) / (maxZ - minZ)));
      const w = Math.max(0, Math.min(1, (this.w - minW) / (maxW - minW)));
      return new Vec4(x * x * (3 - 2 * x), y * y * (3 - 2 * y),
                      z * z * (3 - 2 * z), w * w * (3 - 2 * w));
    }
    toString() { return this.x + ' ' + this.y + ' ' + this.z + ' ' + this.w; }
    toConfigString() { return this.toString(); }
  });
)JS";

} // namespace wallpaper
