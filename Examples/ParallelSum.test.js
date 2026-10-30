var h;
h ||= typeof Module !== 'undefined' ? Module : {};
var aa = !!globalThis.window, ba = !!globalThis.WorkerGlobalScope, k = globalThis.process?.versions?.node && globalThis.process?.type != "renderer", ca = [], l = "./this.program", da = (a, b) => {
  throw b;
}, ea = globalThis.document?.currentScript?.src;
typeof __filename !== "undefined" ? ea = __filename : ba && (ea = self.location.href);
var p = "", ha, u;
if (k) {
  var fs = require("node:fs");
  p = __dirname + "/";
  u = a => {
    a = v(a) ? new URL(a) : a;
    return fs.readFileSync(a);
  };
  ha = async a => {
    a = v(a) ? new URL(a) : a;
    return fs.readFileSync(a, void 0);
  };
  process.argv.length > 1 && (l = process.argv[1].replace(/\\/g, "/"));
  ca = process.argv.slice(2);
  typeof module !== "undefined" && (module.exports = h);
  da = (a, b) => {
    process.exitCode = a;
    throw b;
  };
} else if (aa || ba) {
  try {
    p = (new URL(".", ea)).href;
  } catch {
  }
  ba && (u = a => {
    var b = new XMLHttpRequest();
    b.open("GET", a, !1);
    b.responseType = "arraybuffer";
    b.send(null);
    return new Uint8Array(b.response);
  });
  ha = async a => {
    if (v(a)) {
      return new Promise((c, d) => {
        var e = new XMLHttpRequest();
        e.open("GET", a, !0);
        e.responseType = "arraybuffer";
        e.onload = () => {
          e.status == 200 || e.status == 0 && e.response ? c(e.response) : d(e.status);
        };
        e.onerror = d;
        e.send(null);
      });
    }
    var b = await fetch(a, {credentials:"same-origin"});
    if (b.ok) {
      return b.arrayBuffer();
    }
    throw Error(b.status + " : " + b.url);
  };
}
var ia = console.log.bind(console), w = console.error.bind(console), x = !1, y, v = a => a.startsWith("file://");
function ja() {
  if (!z?.buffer?.resizable) {
    var a = A.buffer;
    z = new Int8Array(a);
    B = new Int16Array(a);
    D = new Uint8Array(a);
    E = new Int32Array(a);
    F = new Uint32Array(a);
    ka = new BigInt64Array(a);
  }
}
function la() {
  var a = h.preRun;
  a && (typeof a === "function" && (a = [a]), ma.push(...a));
  for (a = ma; a.length > 0;) {
    a.shift()(h);
  }
}
function na() {
  var a = h.postRun;
  a && (typeof a === "function" && (a = [a]), oa.push(...a));
  for (a = oa; a.length > 0;) {
    a.shift()(h);
  }
}
function G(a) {
  h.onAbort?.(a);
  a = `Aborted(${a})`;
  w(a);
  x = !0;
  throw new WebAssembly.RuntimeError(a + ". Build with -sASSERTIONS for more info.");
}
var pa;
async function qa(a) {
  try {
    var b = await ha(a);
    return new Uint8Array(b);
  } catch {
  }
  if (u) {
    a = u(a);
  } else {
    throw "both async and sync fetching of the wasm failed";
  }
  return a;
}
async function ra(a, b) {
  try {
    var c = await qa(a);
    return await WebAssembly.instantiate(c, b);
  } catch (d) {
    w(`failed to asynchronously prepare wasm: ${d}`), G(d);
  }
}
async function sa(a) {
  var b = pa;
  if (!v(b) && !k) {
    try {
      var c = fetch(b, {credentials:"same-origin"});
      return await WebAssembly.instantiateStreaming(c, a);
    } catch (d) {
      w(`wasm streaming compile failed: ${d}`), w("falling back to ArrayBuffer instantiation");
    }
  }
  return ra(b, a);
}
class ta {
  name = "ExitStatus";
  constructor(a) {
    this.message = `Program terminated with exit(${a})`;
    this.status = a;
  }
}
var z, oa = [], ma = [], ua = !0, E, I = () => {
  var a = E[+H >>> 2 >>> 0];
  H += 4;
  return a;
}, va = (a, b) => {
  for (var c = 0, d = a.length - 1; d >= 0; d--) {
    var e = a[d];
    e === "." ? a.splice(d, 1) : e === ".." ? (a.splice(d, 1), c++) : c && (a.splice(d, 1), c--);
  }
  if (b) {
    for (; c; c--) {
      a.unshift("..");
    }
  }
  return a;
}, wa = a => {
  var b = a.charAt(0) === "/", c = a.slice(-1) === "/";
  (a = va(a.split("/").filter(d => !!d), !b).join("/")) || b || (a = ".");
  a && c && (a += "/");
  return (b ? "/" : "") + a;
}, xa = a => {
  var b = /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/.exec(a).slice(1);
  a = b[0];
  b = b[1];
  if (!a && !b) {
    return ".";
  }
  b &&= b.slice(0, -1);
  return a + b;
}, ya = () => {
  if (k) {
    var a = require("node:crypto");
    return b => (a.randomFillSync(b), 0);
  }
  return b => (crypto.getRandomValues(b), 0);
}, za = a => (za = ya())(a), Aa = (...a) => {
  for (var b = "", c = !1, d = a.length - 1; d >= -1 && !c; d--) {
    c = d >= 0 ? a[d] : "/";
    if (typeof c !== "string") {
      throw new TypeError("Arguments to path.resolve must be strings");
    }
    if (!c) {
      return "";
    }
    b = c + "/" + b;
    c = c.charAt(0) === "/";
  }
  b = va(b.split("/").filter(e => !!e), !c).join("/");
  return (c ? "/" : "") + b || ".";
}, Ba = globalThis.TextDecoder && new TextDecoder(), J = (a, b = 0) => {
  b >>>= 0;
  var c = b;
  for (var d = c + NaN; a[c] && !(c >= d);) {
    ++c;
  }
  if (c - b > 16 && a.buffer && Ba) {
    return Ba.decode(a.subarray(b, c));
  }
  for (d = ""; b < c;) {
    var e = a[b++];
    if (e & 128) {
      var f = a[b++] & 63;
      if ((e & 224) == 192) {
        d += String.fromCharCode((e & 31) << 6 | f);
      } else {
        var g = a[b++] & 63;
        e = (e & 240) == 224 ? (e & 15) << 12 | f << 6 | g : (e & 7) << 18 | f << 12 | g << 6 | a[b++] & 63;
        e < 65536 ? d += String.fromCharCode(e) : (e -= 65536, d += String.fromCharCode(55296 | e >> 10, 56320 | e & 1023));
      }
    } else {
      d += String.fromCharCode(e);
    }
  }
  return d;
}, Ca = [], Da = a => {
  for (var b = 0, c = 0; c < a.length; ++c) {
    var d = a.charCodeAt(c);
    d <= 127 ? b++ : d <= 2047 ? b += 2 : d >= 55296 && d <= 57343 ? (b += 4, ++c) : b += 3;
  }
  return b;
}, Ea = (a, b, c, d) => {
  c >>>= 0;
  if (!(d > 0)) {
    return 0;
  }
  var e = c;
  d = c + d - 1;
  for (var f = 0; f < a.length; ++f) {
    var g = a.codePointAt(f);
    if (g <= 127) {
      if (c >= d) {
        break;
      }
      b[c++ >>> 0] = g;
    } else if (g <= 2047) {
      if (c + 1 >= d) {
        break;
      }
      b[c++ >>> 0] = 192 | g >> 6;
      b[c++ >>> 0] = 128 | g & 63;
    } else if (g <= 65535) {
      if (c + 2 >= d) {
        break;
      }
      b[c++ >>> 0] = 224 | g >> 12;
      b[c++ >>> 0] = 128 | g >> 6 & 63;
      b[c++ >>> 0] = 128 | g & 63;
    } else {
      if (c + 3 >= d) {
        break;
      }
      b[c++ >>> 0] = 240 | g >> 18;
      b[c++ >>> 0] = 128 | g >> 12 & 63;
      b[c++ >>> 0] = 128 | g >> 6 & 63;
      b[c++ >>> 0] = 128 | g & 63;
      f++;
    }
  }
  b[c >>> 0] = 0;
  return c - e;
}, Fa = [];
function Ga(a, b) {
  Fa[a] = {input:[], output:[], A:b};
  Ha(a, Ia);
}
var Ia = {open(a) {
  var b = Fa[a.node.rdev];
  if (!b) {
    throw new K(43);
  }
  a.tty = b;
  a.seekable = !1;
}, close(a) {
  a.tty.A.fsync(a.tty);
}, fsync(a) {
  a.tty.A.fsync(a.tty);
}, read(a, b, c, d) {
  if (!a.tty || !a.tty.A.M) {
    throw new K(60);
  }
  for (var e = 0, f = 0; f < d; f++) {
    try {
      var g = a.tty.A.M(a.tty);
    } catch (m) {
      throw new K(29);
    }
    if (g === void 0 && !e) {
      throw new K(6);
    }
    if (g === null || g === void 0) {
      break;
    }
    e++;
    b[c + f] = g;
    if (g === 10) {
      break;
    }
  }
  e && (a.node.atime = Date.now());
  return e;
}, write(a, b, c, d) {
  if (!a.tty || !a.tty.A.K) {
    throw new K(60);
  }
  try {
    for (var e = 0; e < d; e++) {
      a.tty.A.K(a.tty, b[c + e]);
    }
  } catch (f) {
    throw new K(29);
  }
  d && (a.node.mtime = a.node.ctime = Date.now());
  return e;
}}, Ja = {M() {
  a: {
    if (!Ca.length) {
      var a = null;
      if (k) {
        var b = Buffer.alloc(256), c = 0;
        try {
          c = fs.readSync(process.stdin.fd, b, 0, 256);
        } catch (d) {
          if (d.toString().includes("EOF")) {
            c = 0;
          } else {
            throw d;
          }
        }
        c > 0 && (a = b.toString("utf-8", 0, c));
      } else {
        globalThis.window?.prompt && (a = window.prompt("Input: "), a !== null && (a += "\n"));
      }
      if (!a) {
        a = null;
        break a;
      }
      b = Array(Da(a) + 1);
      a = Ea(a, b, 0, b.length);
      b.length = a;
      Ca = b;
    }
    a = Ca.shift();
  }
  return a;
}, K(a, b) {
  b === null || b === 10 ? (ia(J(a.output)), a.output = []) : b != 0 && a.output.push(b);
}, fsync(a) {
  a.output?.length > 0 && (ia(J(a.output)), a.output = []);
}, X() {
  return {ia:25856, ka:5, ha:191, ja:35387, ga:[3, 28, 127, 21, 4, 0, 1, 0, 17, 19, 26, 0, 18, 15, 23, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]};
}, Y() {
  return 0;
}, Z() {
  return [24, 80];
}}, Ka = {K(a, b) {
  b === null || b === 10 ? (w(J(a.output)), a.output = []) : b != 0 && a.output.push(b);
}, fsync(a) {
  a.output?.length > 0 && (w(J(a.output)), a.output = []);
}}, L = {u:null, v() {
  return L.createNode(null, "/", 16895, 0);
}, createNode(a, b, c, d) {
  if ((c & 61440) === 24576 || (c & 61440) === 4096) {
    throw new K(63);
  }
  L.u || (L.u = {dir:{node:{C:L.i.C, B:L.i.B, lookup:L.i.lookup, D:L.i.D, rename:L.i.rename, unlink:L.i.unlink, rmdir:L.i.rmdir, readdir:L.i.readdir, symlink:L.i.symlink}, stream:{o:L.h.o}}, file:{node:{C:L.i.C, B:L.i.B}, stream:{o:L.h.o, read:L.h.read, write:L.h.write, P:L.h.P, S:L.h.S}}, link:{node:{C:L.i.C, B:L.i.B, readlink:L.i.readlink}, stream:{}}, L:{node:{C:L.i.C, B:L.i.B}, stream:La}});
  c = Ma(a, b, c, d);
  (c.mode & 61440) === 16384 ? (c.i = L.u.dir.node, c.h = L.u.dir.stream, c.j = {}) : (c.mode & 61440) === 32768 ? (c.i = L.u.file.node, c.h = L.u.file.stream, c.l = 0, c.j = L.U ?? (L.U = new Uint8Array(0))) : (c.mode & 61440) === 40960 ? (c.i = L.u.link.node, c.h = L.u.link.stream) : (c.mode & 61440) === 8192 && (c.i = L.u.L.node, c.h = L.u.L.stream);
  c.atime = c.mtime = c.ctime = Date.now();
  a && (a.j[b] = c, a.atime = a.mtime = a.ctime = c.atime);
  return c;
}, i:{C(a) {
  var b = {};
  b.dev = (a.mode & 61440) === 8192 ? a.id : 1;
  b.ino = a.id;
  b.mode = a.mode;
  b.nlink = 1;
  b.uid = 0;
  b.gid = 0;
  b.rdev = a.rdev;
  (a.mode & 61440) === 16384 ? b.size = 4096 : (a.mode & 61440) === 32768 ? b.size = a.l : (a.mode & 61440) === 40960 ? b.size = a.link.length : b.size = 0;
  b.atime = new Date(a.atime);
  b.mtime = new Date(a.mtime);
  b.ctime = new Date(a.ctime);
  b.blksize = 4096;
  b.blocks = Math.ceil(b.size / b.blksize);
  return b;
}, B(a, b) {
  for (var c of ["mode", "atime", "mtime", "ctime"]) {
    b[c] != null && (a[c] = b[c]);
  }
  b.size !== void 0 && (b = b.size, a.l != b && (c = a.j, a.j = new Uint8Array(b), a.j.set(c.subarray(0, Math.min(b, a.l))), a.l = b));
}, lookup() {
  L.G || (L.G = new K(44), L.G.stack = "<generic error, no stack>");
  throw L.G;
}, D(a, b, c, d) {
  return L.createNode(a, b, c, d);
}, rename(a, b, c) {
  try {
    var d = M(b, c);
  } catch (f) {
  }
  if (d && (a.mode & 61440) === 16384) {
    for (var e in d.j) {
      throw new K(55);
    }
  }
  delete a.parent.j[a.name];
  b.j[c] = a;
  a.name = c;
  b.ctime = b.mtime = a.parent.ctime = a.parent.mtime = Date.now();
}, unlink(a, b) {
  delete a.j[b];
  a.ctime = a.mtime = Date.now();
}, rmdir(a, b) {
  var c = M(a, b), d;
  for (d in c.j) {
    throw new K(55);
  }
  delete a.j[b];
  a.ctime = a.mtime = Date.now();
}, readdir(a) {
  return [".", "..", ...Object.keys(a.j)];
}, symlink(a, b, c) {
  a = L.createNode(a, b, 41471, 0);
  a.link = c;
  return a;
}, readlink(a) {
  if ((a.mode & 61440) !== 40960) {
    throw new K(28);
  }
  return a.link;
}}, h:{read(a, b, c, d, e) {
  if (e >= a.node.l) {
    return 0;
  }
  d = Math.min(a.node.l - e, d);
  b.set(a.node.j.subarray(e, e + d), c);
  return d;
}, write(a, b, c, d, e, f) {
  b.buffer === z.buffer && (f = !1);
  if (!d) {
    return 0;
  }
  a = a.node;
  a.mtime = a.ctime = Date.now();
  if (f) {
    a.j = b.subarray(c, c + d), a.l = d;
  } else if (a.l || e) {
    f = e + d;
    var g = a.j.length;
    g >= f || (f = Math.max(f, g * (g < 1048576 ? 2 : 1.125) >>> 0), g && (f = Math.max(f, 256)), g = a.j.subarray(0, a.l), a.j = new Uint8Array(f), a.j.set(g));
    a.j.set(b.subarray(c, c + d), e);
    a.l = Math.max(a.l, e + d);
  } else {
    a.j = b.slice(c, c + d), a.l = d;
  }
  return d;
}, o(a, b, c) {
  c === 1 ? b += a.position : c === 2 && (a.node.mode & 61440) === 32768 && (b += a.node.l);
  if (b < 0) {
    throw new K(28);
  }
  return b;
}, P(a, b, c, d, e) {
  if ((a.node.mode & 61440) !== 32768) {
    throw new K(43);
  }
  a = a.node.j;
  if (e & 2 || a.buffer !== z.buffer) {
    d = !0;
    G();
    e = void 0;
    if (!e) {
      throw new K(48);
    }
    if (a) {
      if (c > 0 || c + b < a.length) {
        a.subarray ? a = a.subarray(c, c + b) : a = Array.prototype.slice.call(a, c, c + b);
      }
      z.set(a, e >>> 0);
    }
  } else {
    d = !1, e = a.byteOffset;
  }
  return {oa:e, fa:d};
}, S(a, b, c, d) {
  L.h.write(a, b, 0, d, c, !1);
  return 0;
}}}, Na = (a, b) => {
  var c = 0;
  a && (c |= 365);
  b && (c |= 146);
  return c;
}, Oa = null, Pa = {}, N = [], Qa = 1, O = [], Ra = !1, Sa = !0, K = class {
  name = "ErrnoError";
  constructor(a) {
    this.m = a;
  }
}, Ta = class {
  shared = {};
  node = null;
  get flags() {
    return this.shared.flags;
  }
  set flags(a) {
    this.shared.flags = a;
  }
  get position() {
    return this.shared.position;
  }
  set position(a) {
    this.shared.position = a;
  }
}, Ua = class {
  i = {};
  h = {};
  F = null;
  listeners = null;
  N = 0;
  constructor(a, b, c, d) {
    a ||= this;
    this.parent = a;
    this.v = a.v;
    this.id = Qa++;
    this.name = b;
    this.mode = c;
    this.rdev = d;
    this.atime = this.mtime = this.ctime = Date.now();
  }
  get read() {
    return (this.mode & 365) === 365;
  }
  set read(a) {
    a ? this.mode |= 365 : this.mode &= -366;
  }
  get write() {
    return (this.mode & 146) === 146;
  }
  set write(a) {
    a ? this.mode |= 146 : this.mode &= -147;
  }
  da() {
    if (this.listeners) {
      var a;
      for (b of this.listeners) {
        b.exclusive ? (a ||= []).push(b) : b.T(32);
      }
      if (a) {
        var b = (this.N || 0) % a.length;
        this.N = b + 1;
        a[b].T(32);
      }
    }
  }
};
function Q(a, b = {}) {
  if (!a) {
    throw new K(44);
  }
  b.I ?? (b.I = !0);
  a.charAt(0) === "/" || (a = "//" + a);
  var c = 0;
  a: for (; c < 40; c++) {
    a = a.split("/").filter(m => !!m);
    for (var d = Oa, e = "/", f = 0; f < a.length; f++) {
      var g = f === a.length - 1;
      if (g && b.parent) {
        break;
      }
      if (a[f] !== ".") {
        if (a[f] === "..") {
          if (e = xa(e), d === d.parent) {
            a = e + "/" + a.slice(f + 1).join("/");
            c--;
            continue a;
          } else {
            d = d.parent;
          }
        } else {
          e = wa(e + "/" + a[f]);
          try {
            d = M(d, a[f]);
          } catch (m) {
            if (m?.m === 44 && g && b.ba) {
              return {path:e};
            }
            throw m;
          }
          !d.F || g && !b.I || (d = d.F.root);
          if ((d.mode & 61440) === 40960 && (!g || b.H)) {
            if (!d.i.readlink) {
              throw new K(52);
            }
            d = d.i.readlink(d);
            d.charAt(0) === "/" || (d = xa(e) + "/" + d);
            a = d + "/" + a.slice(f + 1).join("/");
            continue a;
          }
        }
      }
    }
    return {path:e, node:d};
  }
  throw new K(32);
}
function Va(a, b) {
  for (var c = 0, d = 0; d < b.length; d++) {
    c = (c << 5) - c + b.charCodeAt(d) | 0;
  }
  return (a + c >>> 0) % O.length;
}
function M(a, b) {
  var c = (a.mode & 61440) === 16384 ? (c = Wa(a, "x")) ? c : a.i.lookup ? 0 : 2 : 54;
  if (c) {
    throw new K(c);
  }
  for (c = O[Va(a.id, b)]; c; c = c.aa) {
    var d = c.name;
    if (c.parent.id === a.id && d === b) {
      return c;
    }
  }
  return a.i.lookup(a, b);
}
function Ma(a, b, c, d) {
  a = new Ua(a, b, c, d);
  b = Va(a.parent.id, a.name);
  a.aa = O[b];
  return O[b] = a;
}
function Wa(a, b) {
  return Sa ? 0 : b.includes("r") && !(a.mode & 292) || b.includes("w") && !(a.mode & 146) || b.includes("x") && !(a.mode & 73) ? 2 : 0;
}
function Xa(a, b) {
  if ((a.mode & 61440) !== 16384) {
    return 54;
  }
  try {
    return M(a, b), 20;
  } catch (c) {
  }
  return Wa(a, "wx");
}
function R(a) {
  a = N[a];
  if (!a) {
    throw new K(8);
  }
  return a;
}
function Ya(a, b = -1) {
  a = Object.assign(new Ta(), a);
  if (b == -1) {
    a: {
      for (b = 0; b <= 4096; b++) {
        if (!N[b]) {
          break a;
        }
      }
      throw new K(33);
    }
  }
  a.fd = b;
  return N[b] = a;
}
function Za(a, b = -1) {
  a = Ya(a, b);
  a.h?.ma?.(a);
  return a;
}
function $a(a, b) {
  var c = void 0, d = c ? null : a;
  c ??= a.i.B;
  if (!c) {
    throw new K(63);
  }
  try {
    c(d, b);
  } catch (e) {
    if (e instanceof RangeError) {
      throw new K(22);
    }
    throw e;
  }
}
var La = {open(a) {
  a.h = Pa[a.node.rdev].h;
  a.h.open?.(a);
}, o() {
  throw new K(70);
}};
function Ha(a, b) {
  Pa[a] = {h:b};
}
function ab(a, b) {
  var c = b === "/";
  if (c && Oa) {
    throw new K(10);
  }
  if (!c && b) {
    var d = Q(b, {I:!1});
    b = d.path;
    d = d.node;
    if (d.F) {
      throw new K(10);
    }
    if ((d.mode & 61440) !== 16384) {
      throw new K(54);
    }
  }
  b = {type:a, na:{}, R:b, $:[]};
  a = a.v(b);
  a.v = b;
  b.root = a;
  c ? Oa = a : d && (d.F = b, d.v && d.v.$.push(b));
}
function bb(a, b, c) {
  var d = Q(a, {parent:!0}).node;
  a = a && a.match(/([^\/]+|\/)\/*$/)[1];
  if (!a) {
    throw new K(28);
  }
  if (a === "." || a === "..") {
    throw new K(20);
  }
  var e = Xa(d, a);
  if (e) {
    throw new K(e);
  }
  if (!d.i.D) {
    throw new K(63);
  }
  return d.i.D(d, a, b, c);
}
function S(a) {
  return bb(a, 16895, 0);
}
function cb(a, b, c) {
  typeof c === "undefined" && (c = b, b = 438);
  bb(a, b | 8192, c);
}
function db(a, b) {
  if (!Aa(a)) {
    throw new K(44);
  }
  var c = Q(b, {parent:!0}).node;
  if (!c) {
    throw new K(44);
  }
  b = b && b.match(/([^\/]+|\/)\/*$/)[1];
  var d = Xa(c, b);
  if (d) {
    throw new K(d);
  }
  if (!c.i.symlink) {
    throw new K(63);
  }
  c.i.symlink(c, b, a);
}
function eb(a, b, c = 438) {
  if (a === "") {
    throw new K(44);
  }
  if (typeof b === "string") {
    var d = {r:0, "r+":2, w:577, "w+":578, a:1089, "a+":1090}[b];
    if (typeof d === "undefined") {
      throw Error(`Unknown file open mode: ${b}`);
    }
    b = d;
  }
  c = b & 64 ? c & 4095 | 32768 : 0;
  if (typeof a === "object") {
    d = a;
  } else {
    var e = a.endsWith("/");
    var f = Q(a, {H:!(b & 131072), ba:!0});
    d = f.node;
    a = f.path;
  }
  f = !1;
  if (b & 64) {
    if (d) {
      if (b & 128) {
        throw new K(20);
      }
    } else {
      if (e) {
        throw new K(31);
      }
      d = bb(a, c | 511, 0);
      f = !0;
    }
  }
  if (!d) {
    throw new K(44);
  }
  (d.mode & 61440) === 8192 && (b &= -513);
  if (b & 65536 && (d.mode & 61440) !== 16384) {
    throw new K(54);
  }
  if (!f && (d ? (d.mode & 61440) === 40960 ? e = 32 : (e = ["r", "w", "rw"][b & 3], b & 512 && (e += "w"), e = (d.mode & 61440) === 16384 && (e !== "r" || b & 576) ? 31 : Wa(d, e)) : e = 44, e)) {
    throw new K(e);
  }
  if (b & 512 && !f) {
    e = d;
    e = typeof e === "string" ? Q(e, {H:!0}).node : e;
    if ((e.mode & 61440) === 16384) {
      throw new K(31);
    }
    if ((e.mode & 61440) !== 32768) {
      throw new K(28);
    }
    if (a = Wa(e, "w")) {
      throw new K(a);
    }
    $a(e, {size:0, timestamp:Date.now()});
  }
  a: {
    for (e = d;;) {
      if (e === e.parent) {
        e = e.v.R;
        var g = g ? e[e.length - 1] !== "/" ? `${e}/${g}` : e + g : e;
        break a;
      }
      g = g ? `${e.name}/${g}` : e.name;
      e = e.parent;
    }
    g = void 0;
  }
  b = Ya({node:d, path:g, flags:b & -131713, seekable:!0, position:0, h:d.h, ea:[], error:!1});
  b.h.open && b.h.open(b);
  f && (c &= 511, d = typeof d === "string" ? Q(d, {H:!0}).node : d, $a(d, {mode:c & 4095 | d.mode & -4096, ctime:Date.now(), la:!1}));
  return b;
}
function fb(a, b, c) {
  if (a.fd === null) {
    throw new K(8);
  }
  if (!a.seekable || !a.h.o) {
    throw new K(70);
  }
  if (c != 0 && c != 1 && c != 2) {
    throw new K(28);
  }
  a.position = a.h.o(a, b, c);
  a.ea = [];
}
function gb(a, b, c, d) {
  var e = void 0;
  if (d < 0 || e < 0) {
    throw new K(28);
  }
  if (a.fd === null) {
    throw new K(8);
  }
  if ((a.flags & 2097155) === 0) {
    throw new K(8);
  }
  if ((a.node.mode & 61440) === 16384) {
    throw new K(31);
  }
  if (!a.h.write) {
    throw new K(28);
  }
  a.seekable && a.flags & 1024 && fb(a, 0, 2);
  var f = typeof e !== "undefined";
  if (!f) {
    e = a.position;
  } else if (!a.seekable) {
    throw new K(70);
  }
  b = a.h.write(a, b, c, d, e, void 0);
  f || (a.position += b);
  return b;
}
function hb() {
  var a, b, c;
  Ra = !0;
  a ??= h.stdin;
  b ??= h.stdout;
  c ??= h.stderr;
  a ? T("stdin", a) : db("/dev/tty", "/dev/stdin");
  b ? T("stdout", null, b) : db("/dev/tty", "/dev/stdout");
  c ? T("stderr", null, c) : db("/dev/tty1", "/dev/stderr");
  eb("/dev/stdin", 0);
  eb("/dev/stdout", 1);
  eb("/dev/stderr", 1);
}
function T(a, b, c) {
  a = wa("/dev/" + a);
  var d = Na(!!b, !!c);
  T.O ?? (T.O = 64);
  var e = T.O++ << 8 | 0;
  Ha(e, {open(f) {
    f.seekable = !1;
  }, close() {
    c?.buffer?.length && c(10);
  }, read(f, g, m, q) {
    for (var n = 0, r = 0; r < q; r++) {
      try {
        var t = b();
      } catch (P) {
        throw new K(29);
      }
      if (t === void 0 && !n) {
        throw new K(6);
      }
      if (t === null || t === void 0) {
        break;
      }
      n++;
      g[m + r] = t;
    }
    n && (f.node.atime = Date.now());
    return n;
  }, write(f, g, m, q) {
    for (var n = 0; n < q; n++) {
      try {
        c(g[m + n]);
      } catch (r) {
        throw new K(29);
      }
    }
    q && (f.node.mtime = f.node.ctime = Date.now());
    return n;
  }});
  cb(a, d, e);
}
var U = {}, D, F, ka, H = void 0, B, ib = {}, kb = () => {
  if (!jb) {
    var a = {USER:"web_user", LOGNAME:"web_user", PATH:"/", PWD:"/", HOME:"/home/web_user", LANG:(globalThis.navigator?.language ?? "C").replace("-", "_") + ".UTF-8", _:l}, b;
    for (b in ib) {
      ib[b] === void 0 ? delete a[b] : a[b] = ib[b];
    }
    var c = [];
    for (b in a) {
      c.push(`${b}=${a[b]}`);
    }
    jb = c;
  }
  return jb;
}, jb, lb = a => {
  y = a;
  ua || (h.onExit?.(a), x = !0);
  da(a, new ta(a));
}, pb = a => {
  try {
    a();
  } catch (b) {
    G(b);
  }
}, qb = a => {
  if (a instanceof ta || a == "unwind") {
    return y;
  }
  da(1, a);
}, rb = a => {
  var b = [];
  if (!x) {
    try {
      return a(...b);
    } catch (c) {
      qb(c);
    } finally {
      if (!ua) {
        try {
          y = a = y, lb(a);
        } catch (c) {
          qb(c);
        }
      }
    }
  }
};
function sb(a) {
  var b = (...c) => {
    V.push(a);
    try {
      return a(...c);
    } finally {
      x || (V.pop(), tb());
    }
  };
  ub.set(a, b);
  return b;
}
var W = 0, X = null, vb = 0, V = [], wb = new Map(), xb = new Map(), ub = new Map(), yb = 0, zb = null, Ab = [];
function tb() {
  X && W === 1 && !V.length && (W = 0, pb(Bb), typeof Fibers !== "undefined" && Fibers.pa());
}
function Cb() {
  var a = Db(1048588), b = a + 12;
  F[a >>> 2 >>> 0] = b;
  F[a + 4 >>> 2 >>> 0] = b + 1048576;
  b = V[0];
  if (!wb.has(b)) {
    var c = yb++;
    wb.set(b, c);
    xb.set(c, b);
  }
  b = wb.get(b);
  E[a + 8 >>> 2 >>> 0] = b;
  return a;
}
function Eb() {
  var a = xb.get(E[X + 8 >>> 2 >>> 0]);
  a = ub.get(a);
  return rb(a);
}
function Fb(a) {
  if (!x) {
    if (W === 0) {
      var b = !1, c = !1;
      a((e = 0) => {
        if (!x && (vb = e, b = !0, c)) {
          W = 2;
          pb(() => Gb(X));
          typeof MainLoop !== "undefined" && MainLoop.V && MainLoop.resume();
          e = !1;
          try {
            var f = Eb();
          } catch (q) {
            f = q, e = !0;
          }
          var g = !1;
          if (!X) {
            var m = zb;
            m && (zb = null, (e ? m.reject : m.resolve)(f), g = !0);
          }
          if (e && !g) {
            throw f;
          }
        }
      });
      c = !0;
      b || (W = 1, X = Cb(), typeof MainLoop !== "undefined" && MainLoop.V && MainLoop.pause(), pb(() => Hb(X)));
    } else if (W === 2) {
      W = 0;
      pb(Ib);
      Jb(X);
      X = null;
      for (var d of Ab) {
        rb(d);
      }
    } else {
      G(`invalid state: ${W}`);
    }
    return vb;
  }
}
O.length = 4096;
ab(L, "/");
S("/tmp");
S("/home");
S("/home/web_user");
(function() {
  S("/dev");
  Ha(259, {read:() => 0, write:(d, e, f, g) => g, o:() => 0});
  cb("/dev/null", 259);
  Ga(1280, Ja);
  Ga(1536, Ka);
  cb("/dev/tty", 1280);
  cb("/dev/tty1", 1536);
  var a = new Uint8Array(1024), b = 0, c = () => {
    b || (za(a), b = a.byteLength);
    return a[--b];
  };
  T("random", c);
  T("urandom", c);
  S("/dev/shm");
  S("/dev/shm/tmp");
})();
(function() {
  S("/proc");
  var a = S("/proc/self");
  S("/proc/self/fd");
  ab({v() {
    var b = Ma(a, "fd", 16895, 73);
    b.h = {o:L.h.o};
    b.i = {lookup(c, d) {
      c = +d;
      var e = R(c);
      c = {parent:null, v:{R:"fake"}, i:{readlink:() => e.path}, id:c + 1};
      return c.parent = c;
    }, readdir() {
      return Array.from(N.entries()).filter(([, c]) => c).map(([c]) => c.toString());
    }};
    return b;
  }}, "/proc/self/fd");
})();
h.noExitRuntime && (ua = h.noExitRuntime);
h.print && (ia = h.print);
h.printErr && (w = h.printErr);
h.arguments && (ca = h.arguments);
h.thisProgram && (l = h.thisProgram);
var Y = h.preInit;
if (Y) {
  for (typeof Y === "function" && (h.preInit = Y = [Y]); Y.length > 0;) {
    Y.shift()();
  }
}
h.callMain = Kb;
var Lb, Db, Jb, Mb, dynCall_vii, Hb, Bb, Gb, Ib, A, Nb = {__syscall_fcntl64:function(a, b, c) {
  H = c >>> 0;
  try {
    var d = R(a);
    switch(b) {
      case 0:
        var e = I();
        if (e < 0) {
          break;
        }
        for (; N[e];) {
          e++;
        }
        return Za(d, e).fd;
      case 1:
      case 2:
        return 0;
      case 3:
        return d.flags;
      case 4:
        return e = I(), d.flags = d.flags & -289793 | e & 289792, 0;
      case 12:
        return e = I(), B[e + 0 >>> 1 >>> 0] = 2, 0;
      case 13:
      case 14:
        return 0;
    }
    return -28;
  } catch (f) {
    if (typeof U === "undefined" || f.name !== "ErrnoError") {
      throw f;
    }
    return -f.m;
  }
}, __syscall_ioctl:function(a, b, c) {
  H = c >>> 0;
  try {
    var d = R(a);
    switch(b) {
      case 21509:
        return d.tty ? 0 : -59;
      case 21505:
        if (!d.tty) {
          return -59;
        }
        if (d.tty.A.X) {
          a = [3, 28, 127, 21, 4, 0, 1, 0, 17, 19, 26, 0, 18, 15, 23, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
          var e = I();
          E[e >>> 2 >>> 0] = 25856;
          E[e + 4 >>> 2 >>> 0] = 5;
          E[e + 8 >>> 2 >>> 0] = 191;
          E[e + 12 >>> 2 >>> 0] = 35387;
          for (var f = 0; f < 32; f++) {
            z[e + f + 17 >>> 0] = a[f] || 0;
          }
        }
        return 0;
      case 21510:
      case 21511:
      case 21512:
        return d.tty ? 0 : -59;
      case 21506:
      case 21507:
      case 21508:
        if (!d.tty) {
          return -59;
        }
        if (d.tty.A.Y) {
          for (e = I(), a = [], f = 0; f < 32; f++) {
            a.push(z[e + f + 17 >>> 0]);
          }
        }
        return 0;
      case 21519:
        if (!d.tty) {
          return -59;
        }
        e = I();
        return E[e >>> 2 >>> 0] = 0;
      case 21520:
        return d.tty ? -28 : -59;
      case 21537:
      case 21531:
        e = I();
        if (!d.h.W) {
          throw new K(59);
        }
        return d.h.W(d, b, e);
      case 21523:
        if (!d.tty) {
          return -59;
        }
        d.tty.A.Z && (f = [24, 80], e = I(), B[e >>> 1 >>> 0] = f[0], B[e + 2 >>> 1 >>> 0] = f[1]);
        return 0;
      case 21524:
        return d.tty ? 0 : -59;
      case 21515:
        return d.tty ? 0 : -59;
      default:
        return -28;
    }
  } catch (g) {
    if (typeof U === "undefined" || g.name !== "ErrnoError") {
      throw g;
    }
    return -g.m;
  }
}, __syscall_openat:function(a, b, c, d) {
  b >>>= 0;
  H = d >>>= 0;
  try {
    var e = b;
    b = (e >>>= 0) ? J(D, e) : "";
    e = b;
    if (e.charAt(0) === "/") {
      b = e;
    } else {
      var f = a === -100 ? "/" : R(a).path;
      if (e.length == 0) {
        throw new K(44);
      }
      b = f + "/" + e;
    }
    var g = d ? I() : 0;
    c & 64 && (g &= -19);
    return eb(b, c, g).fd;
  } catch (m) {
    if (typeof U === "undefined" || m.name !== "ErrnoError") {
      throw m;
    }
    return -m.m;
  }
}, clock_time_get:function(a, b, c) {
  if (!(a >= 0 && a <= 3)) {
    return 28;
  }
  ka[c >>> 0 >>> 3 >>> 0] = BigInt(Math.round((a === 0 ? Date.now() : performance.now()) * 1e6));
  return 0;
}, emscripten_resize_heap:function(a) {
  a >>>= 0;
  var b = D.length;
  if (a > 4294901760) {
    return !1;
  }
  for (var c = 1; c <= 4; c *= 2) {
    var d = b * (1 + .2 / c);
    d = Math.min(d, a + 100663296);
    a: {
      d = (Math.min(4294901760, Math.ceil(Math.max(a, d) / 65536) * 65536) - A.buffer.byteLength + 65535) / 65536 | 0;
      try {
        A.grow(d);
        ja();
        var e = 1;
        break a;
      } catch (f) {
      }
      e = void 0;
    }
    if (e) {
      return !0;
    }
  }
  return !1;
}, environ_get:function(a, b) {
  a >>>= 0;
  b >>>= 0;
  var c = 0, d = 0, e;
  for (e of kb()) {
    var f = b + c;
    F[a + d >>> 2 >>> 0] = f;
    c += Ea(e, D, f, Infinity) + 1;
    d += 4;
  }
  return 0;
}, environ_sizes_get:function(a, b) {
  a >>>= 0;
  b >>>= 0;
  var c = kb();
  F[a >>> 2 >>> 0] = c.length;
  a = 0;
  for (var d of c) {
    a += Da(d) + 1;
  }
  F[b >>> 2 >>> 0] = a;
  return 0;
}, exit:a => {
  y = a;
  lb(a);
}, fd_close:function(a) {
  try {
    var b = R(a);
    if (b.fd === null) {
      throw new K(8);
    }
    b.J && (b.J = null);
    b.node?.da();
    try {
      b.h.close && b.h.close(b);
    } catch (c) {
      throw c;
    } finally {
      N[b.fd] = null;
    }
    b.fd = null;
    return 0;
  } catch (c) {
    if (typeof U === "undefined" || c.name !== "ErrnoError") {
      throw c;
    }
    return c.m;
  }
}, fd_read:function(a, b, c, d) {
  b >>>= 0;
  c >>>= 0;
  d >>>= 0;
  try {
    a: {
      var e = R(a);
      a = b;
      for (var f, g = b = 0; g < c; g++) {
        var m = F[a >>> 2 >>> 0], q = F[a + 4 >>> 2 >>> 0];
        a += 8;
        try {
          var n = e, r = m, t = q, P = f, Ob = z;
          if (t < 0 || P < 0) {
            throw new K(28);
          }
          if (n.fd === null) {
            throw new K(8);
          }
          if ((n.flags & 2097155) === 1) {
            throw new K(8);
          }
          if ((n.node.mode & 61440) === 16384) {
            throw new K(31);
          }
          if (!n.h.read) {
            throw new K(28);
          }
          var mb = typeof P !== "undefined";
          if (!mb) {
            P = n.position;
          } else if (!n.seekable) {
            throw new K(70);
          }
          var nb = n.h.read(n, Ob, r, t, P);
          mb || (n.position += nb);
          var fa = nb;
        } catch (C) {
          if (b > 0 && C instanceof K && (C.m == 6 || C.m == 6)) {
            break;
          }
          throw C;
        }
        if (fa < 0) {
          var ob = -1;
          break a;
        }
        b += fa;
        if (fa < q) {
          break;
        }
        typeof f !== "undefined" && (f += fa);
      }
      ob = b;
    }
    F[d >>> 2 >>> 0] = ob;
    return 0;
  } catch (C) {
    if (typeof U === "undefined" || C.name !== "ErrnoError") {
      throw C;
    }
    return C.m;
  }
}, fd_seek:function(a, b, c, d) {
  b = b < -9007199254740992 || b > 9007199254740992 ? NaN : Number(b);
  d >>>= 0;
  try {
    if (isNaN(b)) {
      return 22;
    }
    var e = R(a);
    fb(e, b, c);
    ka[d >>> 3 >>> 0] = BigInt(e.position);
    e.J && !b && c === 0 && (e.J = null);
    return 0;
  } catch (f) {
    if (typeof U === "undefined" || f.name !== "ErrnoError") {
      throw f;
    }
    return f.m;
  }
}, fd_write:function(a, b, c, d) {
  b >>>= 0;
  c >>>= 0;
  d >>>= 0;
  try {
    var e = R(a);
    a = b;
    if (c == 1) {
      var f = gb(e, z, F[a >>> 2 >>> 0], F[a + 4 >>> 2 >>> 0]);
    } else {
      for (var g = b = 0, m = a; g < c; g++, m += 8) {
        b += F[m + 4 >>> 2 >>> 0];
      }
      var q = new Uint8Array(b);
      for (g = m = 0; g < c; g++, a += 8) {
        var n = F[a >>> 2 >>> 0], r = F[a + 4 >>> 2 >>> 0];
        q.set(D.subarray(n >>> 0, n + r >>> 0), m);
        m += r;
      }
      f = gb(e, q, 0, b);
    }
    F[d >>> 2 >>> 0] = f;
    return 0;
  } catch (t) {
    if (typeof U === "undefined" || t.name !== "ErrnoError") {
      throw t;
    }
    return t.m;
  }
}, strict_scan_registers:a => Fb(b => {
  var c = tb;
  tb = () => {
    var d = X && W === 1 && !V.length;
    c();
    d && (tb = c, dynCall_vii(a, X + 12, F[X >>> 2 >>> 0]), b());
  };
})};
function Pb() {
  var a = Z;
  a = Object.assign({}, a);
  var b = c => d => c(d) >>> 0;
  a.malloc = b(a.malloc);
  a._emscripten_stack_alloc = b(a._emscripten_stack_alloc);
  a.emscripten_stack_get_current = (c => () => c() >>> 0)(a.emscripten_stack_get_current);
  return a;
}
function Kb(a = []) {
  var b = Lb;
  a.unshift(l);
  var c = a.length, d = Mb((c + 1) * 4), e = d, f;
  for (f of a) {
    a = F;
    var g = e >>> 2 >>> 0, m = f, q = Da(m) + 1, n = Mb(q);
    Ea(m, D, n, q);
    a[g] = n;
    e += 4;
  }
  F[e >>> 2 >>> 0] = 0;
  try {
    var r = b(c, d);
    y = r;
    lb(r);
    return r;
  } catch (t) {
    return qb(t);
  }
}
async function Qb() {
  var a = ca;
  la();
  var b = h.setStatus;
  b && (b("Running..."), await new Promise(c => setTimeout(c, 1)), setTimeout(b, 1, ""));
  x || (h.noFSInit || Ra || hb(), Z.__wasm_call_ctors(), Sa = !1, h.onRuntimeInitialized?.(), h.noInitialRun || Kb(a), na());
}
var Z;
(async function() {
  function a(d) {
    var e = Z = d.exports;
    d = {};
    for (let [f, g] of Object.entries(e)) {
      typeof g === "function" ? (e = sb(g), d[f] = e) : d[f] = g;
    }
    Z = d;
    d = Z = Pb();
    Lb = h._main = d.main;
    Db = d.malloc;
    Jb = d.free;
    Mb = d._emscripten_stack_alloc;
    dynCall_vii = d.dynCall_vii;
    Hb = d.asyncify_start_unwind;
    Bb = d.asyncify_stop_unwind;
    Gb = d.asyncify_start_rewind;
    Ib = d.asyncify_stop_rewind;
    A = d.memory;
    ja();
    return Z;
  }
  var b = {env:Nb, wasi_snapshot_preview1:Nb}, c = h.instantiateWasm;
  if (c) {
    return new Promise(d => {
      c(b, e => d(a(e)));
    });
  }
  pa ??= h.locateFile ? h.locateFile("ParallelSum.test.wasm", p) : p + "ParallelSum.test.wasm";
  return function(d) {
    return a(d.instance);
  }(await sa(b));
})().then(() => Qb());

