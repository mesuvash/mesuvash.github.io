(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.TransformerViz = {}));
})(this, (function (exports) { 'use strict';

    // Linear algebra module for 3D rendering.
    // All matrices are column-major (index = col * 4 + row), matching WebGL convention.
    // All functions operate on Float32Array. Zero external dependencies.
    // ============================================================================
    // Vec3
    // ============================================================================
    function vec3Create(x, y, z) {
        const out = new Float32Array(3);
        out[0] = x;
        out[1] = y;
        out[2] = z;
        return out;
    }
    // ============================================================================
    // Mat4 — column-major layout
    // ============================================================================
    // Index layout:
    //   Column 0: [0, 1, 2, 3]
    //   Column 1: [4, 5, 6, 7]
    //   Column 2: [8, 9, 10, 11]
    //   Column 3: [12, 13, 14, 15]
    //
    // So the matrix as written on paper:
    //   | m[0]  m[4]  m[8]   m[12] |
    //   | m[1]  m[5]  m[9]   m[13] |
    //   | m[2]  m[6]  m[10]  m[14] |
    //   | m[3]  m[7]  m[11]  m[15] |
    function mat4Create() {
        const out = new Float32Array(16);
        out[0] = 1;
        out[5] = 1;
        out[10] = 1;
        out[15] = 1;
        return out;
    }
    /**
     * Matrix multiply: out = a * b (column-major).
     * If a is a view transform and b is a model transform, the result applies b first, then a.
     */
    function mat4Multiply(out, a, b) {
        const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
        // Column 0 of b
        let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
        out[0] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
        out[1] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
        out[2] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
        out[3] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;
        // Column 1 of b
        b0 = b[4];
        b1 = b[5];
        b2 = b[6];
        b3 = b[7];
        out[4] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
        out[5] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
        out[6] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
        out[7] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;
        // Column 2 of b
        b0 = b[8];
        b1 = b[9];
        b2 = b[10];
        b3 = b[11];
        out[8] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
        out[9] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
        out[10] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
        out[11] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;
        // Column 3 of b
        b0 = b[12];
        b1 = b[13];
        b2 = b[14];
        b3 = b[15];
        out[12] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
        out[13] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
        out[14] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
        out[15] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;
        return out;
    }
    /**
     * Creates a perspective projection matrix.
     *
     * Column-major layout:
     *   | f/aspect  0     0                    0  |
     *   | 0         f     0                    0  |
     *   | 0         0     (far+near)/(near-far) -1 |
     *   | 0         0     2*far*near/(near-far)  0  |
     *
     * where f = 1 / tan(fovy / 2).
     * Maps to clip space z in [-1, 1] (WebGL convention).
     */
    function mat4Perspective(out, fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1.0 / (near - far);
        out[0] = f / aspect;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
        out[4] = 0;
        out[5] = f;
        out[6] = 0;
        out[7] = 0;
        out[8] = 0;
        out[9] = 0;
        out[10] = (far + near) * nf;
        out[11] = -1;
        out[12] = 0;
        out[13] = 0;
        out[14] = 2 * far * near * nf;
        out[15] = 0;
        return out;
    }
    /**
     * Creates a view matrix that looks from `eye` toward `center` with the given `up` direction.
     */
    function mat4LookAt(out, eye, center, up) {
        // Forward vector (camera looks along -z in its local space)
        let fx = eye[0] - center[0];
        let fy = eye[1] - center[1];
        let fz = eye[2] - center[2];
        let len = fx * fx + fy * fy + fz * fz;
        if (len > 0) {
            len = 1.0 / Math.sqrt(len);
            fx *= len;
            fy *= len;
            fz *= len;
        }
        // Right = up x forward
        let rx = up[1] * fz - up[2] * fy;
        let ry = up[2] * fx - up[0] * fz;
        let rz = up[0] * fy - up[1] * fx;
        len = rx * rx + ry * ry + rz * rz;
        if (len > 0) {
            len = 1.0 / Math.sqrt(len);
            rx *= len;
            ry *= len;
            rz *= len;
        }
        else {
            rx = 0;
            ry = 0;
            rz = 0;
        }
        // Recomputed up = forward x right
        let ux = fy * rz - fz * ry;
        let uy = fz * rx - fx * rz;
        let uz = fx * ry - fy * rx;
        len = ux * ux + uy * uy + uz * uz;
        if (len > 0) {
            len = 1.0 / Math.sqrt(len);
            ux *= len;
            uy *= len;
            uz *= len;
        }
        // Column-major: each column is a row of the rotation part, and column 3 is the translation
        out[0] = rx;
        out[1] = ux;
        out[2] = fx;
        out[3] = 0;
        out[4] = ry;
        out[5] = uy;
        out[6] = fy;
        out[7] = 0;
        out[8] = rz;
        out[9] = uz;
        out[10] = fz;
        out[11] = 0;
        out[12] = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
        out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
        out[14] = -(fx * eye[0] + fy * eye[1] + fz * eye[2]);
        out[15] = 1;
        return out;
    }
    /**
     * Composes a transformation matrix from translation (vec3), rotation (quaternion), and scale (vec3).
     * Equivalent to: out = T * R * S
     */
    function mat4FromTranslationRotationScale(out, t, r, s) {
        // Rotation from quaternion
        const x = r[0], y = r[1], z = r[2], w = r[3];
        const x2 = x + x;
        const y2 = y + y;
        const z2 = z + z;
        const xx = x * x2;
        const xy = x * y2;
        const xz = x * z2;
        const yy = y * y2;
        const yz = y * z2;
        const zz = z * z2;
        const wx = w * x2;
        const wy = w * y2;
        const wz = w * z2;
        const sx = s[0], sy = s[1], sz = s[2];
        out[0] = (1 - yy - zz) * sx;
        out[1] = (xy + wz) * sx;
        out[2] = (xz - wy) * sx;
        out[3] = 0;
        out[4] = (xy - wz) * sy;
        out[5] = (1 - xx - zz) * sy;
        out[6] = (yz + wx) * sy;
        out[7] = 0;
        out[8] = (xz + wy) * sz;
        out[9] = (yz - wx) * sz;
        out[10] = (1 - xx - yy) * sz;
        out[11] = 0;
        out[12] = t[0];
        out[13] = t[1];
        out[14] = t[2];
        out[15] = 1;
        return out;
    }
    // ============================================================================
    // Quaternion — stored as [x, y, z, w]
    // ============================================================================
    /**
     * Creates an identity quaternion [0, 0, 0, 1].
     */
    function quatCreate() {
        const out = new Float32Array(4);
        out[3] = 1;
        return out;
    }
    // ============================================================================
    // Utility
    // ============================================================================
    /**
     * Converts degrees to radians.
     */
    function toRadian(degrees) {
        return degrees * (Math.PI / 180);
    }
    /**
     * Projects a 3D point to screen coordinates using a combined view-projection matrix.
     * Returns null if the point is behind the camera (w <= 0).
     * Returns screen-space { x, y } and clip-space z for depth sorting.
     */
    function projectToScreen(point3D, viewProj, viewportWidth, viewportHeight) {
        const x = point3D[0], y = point3D[1], z = point3D[2];
        // Multiply by viewProj (column-major)
        const clipX = viewProj[0] * x + viewProj[4] * y + viewProj[8] * z + viewProj[12];
        const clipY = viewProj[1] * x + viewProj[5] * y + viewProj[9] * z + viewProj[13];
        const clipZ = viewProj[2] * x + viewProj[6] * y + viewProj[10] * z + viewProj[14];
        const clipW = viewProj[3] * x + viewProj[7] * y + viewProj[11] * z + viewProj[15];
        // Behind camera check
        if (clipW <= 0) {
            return null;
        }
        // Perspective divide to NDC [-1, 1]
        const ndcX = clipX / clipW;
        const ndcY = clipY / clipW;
        const ndcZ = clipZ / clipW;
        // NDC to screen coordinates
        // x: [-1, 1] -> [0, viewportWidth]
        // y: [-1, 1] -> [viewportHeight, 0] (screen Y is flipped)
        const screenX = (ndcX + 1) * 0.5 * viewportWidth;
        const screenY = (1 - ndcY) * 0.5 * viewportHeight;
        return { x: screenX, y: screenY, z: ndcZ };
    }

    /**
     * Orbit camera for the 3D transformer visualization.
     * Orbits around a focus point using azimuth/elevation angles.
     */
    const DEFAULT_FOV_Y = 45;
    const DEFAULT_NEAR = 0.1;
    const DEFAULT_FAR = 200;
    const DEFAULT_MIN_DIST = 2;
    const DEFAULT_MAX_DIST = 80;
    const EPSILON = 0.01;
    class OrbitCamera {
        state;
        _fovY;
        _near;
        _far;
        _minDistance;
        _maxDistance;
        _minElevation;
        _maxElevation;
        _viewMatrix;
        _projMatrix;
        _viewProjMatrix;
        _eyePos;
        _aspect = 1;
        _dirty = true;
        constructor(config) {
            this._fovY = toRadian(config?.fovY ?? DEFAULT_FOV_Y);
            this._near = config?.near ?? DEFAULT_NEAR;
            this._far = config?.far ?? DEFAULT_FAR;
            this._minDistance = config?.minDistance ?? DEFAULT_MIN_DIST;
            this._maxDistance = config?.maxDistance ?? DEFAULT_MAX_DIST;
            this._minElevation = config?.minElevation ?? (-Math.PI / 2 + EPSILON);
            this._maxElevation = config?.maxElevation ?? (Math.PI / 2 - EPSILON);
            this.state = {
                azimuth: 0,
                elevation: toRadian(20),
                distance: 25,
                focusPoint: vec3Create(0, 10, 0),
            };
            this._viewMatrix = mat4Create();
            this._projMatrix = mat4Create();
            this._viewProjMatrix = mat4Create();
            this._eyePos = vec3Create(0, 0, 0);
        }
        /** Update the aspect ratio (call on resize). */
        setAspect(width, height) {
            this._aspect = width / height;
            this._dirty = true;
        }
        /** Orbit by delta angles (in radians). */
        orbit(deltaAzimuth, deltaElevation) {
            this.state.azimuth += deltaAzimuth;
            this.state.elevation = Math.max(this._minElevation, Math.min(this._maxElevation, this.state.elevation + deltaElevation));
            this._dirty = true;
        }
        /** Zoom by delta distance (positive = zoom out). */
        zoom(delta) {
            this.state.distance = Math.max(this._minDistance, Math.min(this._maxDistance, this.state.distance + delta));
            this._dirty = true;
        }
        /** Pan the focus point in camera-local XY plane. */
        pan(deltaX, deltaY) {
            const az = this.state.azimuth;
            const el = this.state.elevation;
            // Camera right direction (in world space)
            const rightX = Math.cos(az);
            const rightZ = -Math.sin(az);
            // Camera up direction (in world space, approximate for small elevations)
            const upX = -Math.sin(el) * Math.sin(az);
            const upY = Math.cos(el);
            const upZ = -Math.sin(el) * Math.cos(az);
            // Scale by distance for consistent feel
            const scale = this.state.distance * 0.002;
            this.state.focusPoint[0] += (rightX * deltaX + upX * deltaY) * scale;
            this.state.focusPoint[1] += upY * deltaY * scale;
            this.state.focusPoint[2] += (rightZ * deltaX + upZ * deltaY) * scale;
            this._dirty = true;
        }
        /** Reset to default position. */
        reset() {
            this.state.azimuth = 0;
            this.state.elevation = toRadian(20);
            this.state.distance = 25;
            this.state.focusPoint[0] = 0;
            this.state.focusPoint[1] = 10;
            this.state.focusPoint[2] = 0;
            this._dirty = true;
        }
        /** Get the view matrix. Recomputes if dirty. */
        get viewMatrix() {
            this._updateIfDirty();
            return this._viewMatrix;
        }
        /** Get the projection matrix. Recomputes if dirty. */
        get projMatrix() {
            this._updateIfDirty();
            return this._projMatrix;
        }
        /** Get the combined view-projection matrix. Recomputes if dirty. */
        get viewProjMatrix() {
            this._updateIfDirty();
            return this._viewProjMatrix;
        }
        /** Get the camera eye position in world space. */
        get eyePosition() {
            this._updateIfDirty();
            return this._eyePos;
        }
        _updateIfDirty() {
            if (!this._dirty)
                return;
            this._dirty = false;
            const { azimuth, elevation, distance, focusPoint } = this.state;
            // Compute eye position from spherical coordinates
            const cosEl = Math.cos(elevation);
            this._eyePos[0] = focusPoint[0] + distance * cosEl * Math.sin(azimuth);
            this._eyePos[1] = focusPoint[1] + distance * Math.sin(elevation);
            this._eyePos[2] = focusPoint[2] + distance * cosEl * Math.cos(azimuth);
            const up = vec3Create(0, 1, 0);
            mat4LookAt(this._viewMatrix, this._eyePos, focusPoint, up);
            mat4Perspective(this._projMatrix, this._fovY, this._aspect, this._near, this._far);
            mat4Multiply(this._viewProjMatrix, this._projMatrix, this._viewMatrix);
        }
    }

    /**
     * Scene graph for the 3D visualization.
     * Each SceneNode has a local transform, children, and optional renderable geometry.
     */
    // ─── Factory ─────────────────────────────────────────────────────────
    let _nextPickId = 1;
    function defaultVisualState() {
        return {
            baseColor: [0.5, 0.5, 0.5, 1.0],
            activationMagnitude: 0,
            gradientMagnitude: 0,
            hovered: false,
            selected: false,
            opacity: 1.0,
        };
    }
    /**
     * Create a new SceneNode with default values.
     */
    function createSceneNode(id, pickable = false) {
        return {
            id,
            position: vec3Create(0, 0, 0),
            rotation: quatCreate(),
            scale: vec3Create(1, 1, 1),
            worldMatrix: mat4Create(),
            children: [],
            parent: null,
            renderable: null,
            pickId: pickable ? _nextPickId++ : 0,
            visible: true,
            expanded: false,
            componentSpec: null,
            visualState: defaultVisualState(),
        };
    }
    /** Reset the pick ID counter. Call when rebuilding the entire scene. */
    function resetPickIdCounter() {
        _nextPickId = 1;
    }
    // ─── Tree operations ─────────────────────────────────────────────────
    function addChild(parent, child) {
        if (child.parent) {
            removeChild(child.parent, child);
        }
        child.parent = parent;
        parent.children.push(child);
    }
    function removeChild(parent, child) {
        const idx = parent.children.indexOf(child);
        if (idx >= 0) {
            parent.children.splice(idx, 1);
            child.parent = null;
        }
    }
    // ─── Transform computation ───────────────────────────────────────────
    const _localMat = mat4Create();
    /**
     * Recursively compute world matrices for the entire subtree.
     * Call on the root node each frame before rendering.
     */
    function updateWorldMatrices(node, parentWorld) {
        // Compute local matrix from position/rotation/scale
        mat4FromTranslationRotationScale(_localMat, node.position, node.rotation, node.scale);
        if (parentWorld) {
            mat4Multiply(node.worldMatrix, parentWorld, _localMat);
        }
        else {
            // Root node: world = local
            node.worldMatrix.set(_localMat);
        }
        for (const child of node.children) {
            updateWorldMatrices(child, node.worldMatrix);
        }
    }
    /**
     * Depth-first traversal. Return false from callback to skip children.
     */
    function traverse(node, callback, depth = 0) {
        if (!node.visible)
            return;
        const result = callback(node, depth);
        if (result === false)
            return; // skip children
        for (const child of node.children) {
            traverse(child, callback, depth + 1);
        }
    }
    /**
     * Find a node by its pickId (for color-based picking).
     */
    function findByPickId(root, pickId) {
        let found = null;
        traverse(root, (node) => {
            if (node.pickId === pickId) {
                found = node;
                return false; // stop traversal
            }
        });
        return found;
    }
    /**
     * Set position of a node.
     */
    function setPosition(node, x, y, z) {
        node.position[0] = x;
        node.position[1] = y;
        node.position[2] = z;
    }
    /**
     * Set scale of a node.
     */
    function setScale(node, x, y, z) {
        node.scale[0] = x;
        node.scale[1] = y;
        node.scale[2] = z;
    }

    /**
     * WebGL2 context bootstrap and state management.
     */
    /**
     * Initialize a WebGL2 rendering context inside the given container.
     */
    function createGLContext(container, width, height) {
        const canvas = document.createElement('canvas');
        canvas.style.display = 'block';
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        container.appendChild(canvas);
        const gl = canvas.getContext('webgl2', {
            antialias: true,
            alpha: false,
            premultipliedAlpha: false,
            preserveDrawingBuffer: false,
        });
        if (!gl) {
            throw new Error('[transformer-viz] WebGL2 is not supported in this browser.');
        }
        const programs = new Map();
        const ctx = {
            gl,
            canvas,
            programs,
            resize(w, h) {
                const dpr = window.devicePixelRatio || 1;
                canvas.width = Math.round(w * dpr);
                canvas.height = Math.round(h * dpr);
                canvas.style.width = `${w}px`;
                canvas.style.height = `${h}px`;
                gl.viewport(0, 0, canvas.width, canvas.height);
            },
            destroy() {
                for (const prog of programs.values()) {
                    gl.deleteProgram(prog);
                }
                programs.clear();
                canvas.remove();
            },
        };
        ctx.resize(width, height);
        // Default GL state
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.BACK);
        gl.clearColor(0.96, 0.96, 0.96, 1.0); // #f5f5f5
        return ctx;
    }
    // ─── Shader compilation ──────────────────────────────────────────────
    function compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        if (!shader)
            throw new Error('[transformer-viz] Failed to create shader.');
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const info = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error(`[transformer-viz] Shader compilation failed: ${info}`);
        }
        return shader;
    }
    /**
     * Compile and link a shader program from vertex and fragment source.
     * Stores it in the context's program map under the given name.
     */
    function createProgram(ctx, name, vertSource, fragSource) {
        const { gl } = ctx;
        const vert = compileShader(gl, gl.VERTEX_SHADER, vertSource);
        const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSource);
        const program = gl.createProgram();
        if (!program)
            throw new Error('[transformer-viz] Failed to create program.');
        gl.attachShader(program, vert);
        gl.attachShader(program, frag);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const info = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            gl.deleteShader(vert);
            gl.deleteShader(frag);
            throw new Error(`[transformer-viz] Program link failed: ${info}`);
        }
        // Shaders can be deleted after linking
        gl.deleteShader(vert);
        gl.deleteShader(frag);
        ctx.programs.set(name, program);
        return program;
    }
    /**
     * Get a uniform location, cached per program.
     */
    const _uniformCache = new WeakMap();
    function getUniformLocation(gl, program, name) {
        let cache = _uniformCache.get(program);
        if (!cache) {
            cache = new Map();
            _uniformCache.set(program, cache);
        }
        if (cache.has(name)) {
            return cache.get(name);
        }
        const loc = gl.getUniformLocation(program, name);
        cache.set(name, loc);
        return loc;
    }
    /**
     * Get an attribute location.
     */
    function getAttribLocation(gl, program, name) {
        return gl.getAttribLocation(program, name);
    }

    /**
     * GLSL shader source strings for all shader programs.
     * WebGL2 (GLSL ES 3.00).
     */
    // ─── Solid Program ───────────────────────────────────────────────────
    // Renders geometry with a single color + directional lighting.
    const SOLID_VERT = `#version 300 es
precision highp float;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;

in vec3 aPosition;
in vec3 aNormal;

out vec3 vNormal;
out vec3 vViewPos;

void main() {
  vec4 viewPos = uModelView * vec4(aPosition, 1.0);
  vViewPos = viewPos.xyz;
  vNormal = normalize(uNormalMatrix * aNormal);
  gl_Position = uProjection * viewPos;
}
`;
    const SOLID_FRAG = `#version 300 es
precision highp float;

uniform vec4 uColor;
uniform float uOpacity;

in vec3 vNormal;
in vec3 vViewPos;

out vec4 fragColor;

const vec3 LIGHT_DIR = normalize(vec3(0.3, 0.8, 0.5));
const float AMBIENT = 0.35;
const float DIFFUSE = 0.55;
const float SPECULAR = 0.15;
const float SHININESS = 32.0;

void main() {
  vec3 N = normalize(vNormal);
  vec3 L = LIGHT_DIR;
  vec3 V = normalize(-vViewPos);
  vec3 H = normalize(L + V);

  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(N, H), 0.0), SHININESS);

  // Two-sided lighting: if back-face, flip normal
  if (!gl_FrontFacing) {
    diff = max(dot(-N, L), 0.0);
    spec = pow(max(dot(-N, H), 0.0), SHININESS);
  }

  float light = AMBIENT + DIFFUSE * diff + SPECULAR * spec;
  fragColor = vec4(uColor.rgb * light, uColor.a * uOpacity);
}
`;
    // ─── Gradient-Mapped Program ─────────────────────────────────────────
    // Same as solid but interpolates between base color and gradient color.
    const GRADIENT_VERT = SOLID_VERT; // Same vertex shader
    const GRADIENT_FRAG = `#version 300 es
precision highp float;

uniform vec4 uBaseColor;
uniform vec4 uGradientColor;
uniform float uGradientMix;
uniform float uOpacity;

in vec3 vNormal;
in vec3 vViewPos;

out vec4 fragColor;

const vec3 LIGHT_DIR = normalize(vec3(0.3, 0.8, 0.5));
const float AMBIENT = 0.35;
const float DIFFUSE = 0.55;
const float SPECULAR = 0.15;
const float SHININESS = 32.0;

void main() {
  vec3 N = normalize(vNormal);
  if (!gl_FrontFacing) N = -N;

  vec3 L = LIGHT_DIR;
  vec3 V = normalize(-vViewPos);
  vec3 H = normalize(L + V);

  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(N, H), 0.0), SHININESS);
  float light = AMBIENT + DIFFUSE * diff + SPECULAR * spec;

  vec4 color = mix(uBaseColor, uGradientColor, uGradientMix);
  fragColor = vec4(color.rgb * light, color.a * uOpacity);
}
`;
    // ─── Picking Program ─────────────────────────────────────────────────
    // Flat color render for color-based picking.
    const PICKING_VERT = `#version 300 es
precision highp float;

uniform mat4 uMVP;

in vec3 aPosition;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
}
`;
    const PICKING_FRAG = `#version 300 es
precision highp float;

uniform vec4 uPickColor;

out vec4 fragColor;

void main() {
  fragColor = uPickColor;
}
`;
    // ─── Wireframe Program ───────────────────────────────────────────────
    // Simple flat color with no lighting, for wireframe / outlines.
    const WIREFRAME_VERT = `#version 300 es
precision highp float;

uniform mat4 uMVP;

in vec3 aPosition;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
}
`;
    const WIREFRAME_FRAG = `#version 300 es
precision highp float;

uniform vec4 uColor;

out vec4 fragColor;

void main() {
  fragColor = uColor;
}
`;
    // ─── Particle Program ────────────────────────────────────────────────
    // Point sprites for data flow animation.
    const PARTICLE_VERT = `#version 300 es
precision highp float;

uniform mat4 uMVP;
uniform float uPointSize;

in vec3 aPosition;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
  gl_PointSize = uPointSize;
}
`;
    const PARTICLE_FRAG = `#version 300 es
precision highp float;

uniform vec4 uColor;

out vec4 fragColor;

void main() {
  // Circular point sprite
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);
  if (dist > 0.5) discard;
  // Soft edge
  float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
  fragColor = vec4(uColor.rgb, uColor.a * alpha);
}
`;
    // ─── Arrow/Connection Line Program ───────────────────────────────────
    // Simple colored line with slight alpha.
    const LINE_VERT = `#version 300 es
precision highp float;

uniform mat4 uMVP;

in vec3 aPosition;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
}
`;
    const LINE_FRAG = `#version 300 es
precision highp float;

uniform vec4 uColor;

out vec4 fragColor;

void main() {
  fragColor = uColor;
}
`;

    /**
     * Initializes all shader programs for the renderer.
     */
    /** Program name constants. */
    const PROGRAM = {
        SOLID: 'solid',
        GRADIENT: 'gradient',
        PICKING: 'picking',
        WIREFRAME: 'wireframe',
        PARTICLE: 'particle',
        LINE: 'line',
    };
    /**
     * Compile and register all shader programs.
     */
    function initAllPrograms(ctx) {
        createProgram(ctx, PROGRAM.SOLID, SOLID_VERT, SOLID_FRAG);
        createProgram(ctx, PROGRAM.GRADIENT, GRADIENT_VERT, GRADIENT_FRAG);
        createProgram(ctx, PROGRAM.PICKING, PICKING_VERT, PICKING_FRAG);
        createProgram(ctx, PROGRAM.WIREFRAME, WIREFRAME_VERT, WIREFRAME_FRAG);
        createProgram(ctx, PROGRAM.PARTICLE, PARTICLE_VERT, PARTICLE_FRAG);
        createProgram(ctx, PROGRAM.LINE, LINE_VERT, LINE_FRAG);
    }

    /**
     * Geometry generators for 3D primitives.
     * All geometry is stored in WebGL VAOs for efficient rendering.
     */
    /**
     * Create all primitive geometries. Call once at init.
     */
    function createPrimitives(gl, solidProgram) {
        return {
            box: createBox(gl, solidProgram),
            diamond: createDiamond(gl, solidProgram),
            cylinder: createCylinder(gl, solidProgram, 0.5, 1.0, 16),
            plane: createPlane(gl, solidProgram),
        };
    }
    // ─── Box (unit cube centered at origin) ──────────────────────────────
    function createBox(gl, program) {
        // Unit cube: -0.5 to 0.5 on each axis. Scaled via model matrix.
        const positions = new Float32Array([
            // Front face (z = +0.5)
            -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
            // Back face (z = -0.5)
            -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
            // Top face (y = +0.5)
            -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
            // Bottom face (y = -0.5)
            -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5,
            // Right face (x = +0.5)
            0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5,
            // Left face (x = -0.5)
            -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5,
        ]);
        const normals = new Float32Array([
            // Front
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
            // Back
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            // Top
            0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
            // Bottom
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
            // Right
            1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
            // Left
            -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        ]);
        const indices = new Uint16Array([
            0, 1, 2, 0, 2, 3, // Front
            4, 5, 6, 4, 6, 7, // Back
            8, 9, 10, 8, 10, 11, // Top
            12, 13, 14, 12, 14, 15, // Bottom
            16, 17, 18, 16, 18, 19, // Right
            20, 21, 22, 20, 22, 23, // Left
        ]);
        return createGeometry(gl, program, positions, normals, indices, gl.TRIANGLES);
    }
    // ─── Diamond (octahedron, for residual add nodes) ────────────────────
    function createDiamond(gl, program) {
        // Octahedron vertices
        const t = 0.5;
        const positions = new Float32Array([
            // Top pyramid (4 triangles)
            0, t, 0, t, 0, 0, 0, 0, t, // front-right
            0, t, 0, 0, 0, t, -t, 0, 0, // front-left
            0, t, 0, -t, 0, 0, 0, 0, -t, // back-left
            0, t, 0, 0, 0, -t, t, 0, 0, // back-right
            // Bottom pyramid (4 triangles)
            0, -t, 0, 0, 0, t, t, 0, 0, // front-right
            0, -t, 0, -t, 0, 0, 0, 0, t, // front-left
            0, -t, 0, 0, 0, -t, -t, 0, 0, // back-left
            0, -t, 0, t, 0, 0, 0, 0, -t, // back-right
        ]);
        // Compute face normals for each triangle
        const normals = new Float32Array(positions.length);
        for (let i = 0; i < positions.length; i += 9) {
            const ax = positions[i + 3] - positions[i], ay = positions[i + 4] - positions[i + 1], az = positions[i + 5] - positions[i + 2];
            const bx = positions[i + 6] - positions[i], by = positions[i + 7] - positions[i + 1], bz = positions[i + 8] - positions[i + 2];
            let nx = ay * bz - az * by;
            let ny = az * bx - ax * bz;
            let nz = ax * by - ay * bx;
            const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 0) {
                nx /= len;
                ny /= len;
                nz /= len;
            }
            for (let j = 0; j < 3; j++) {
                normals[i + j * 3] = nx;
                normals[i + j * 3 + 1] = ny;
                normals[i + j * 3 + 2] = nz;
            }
        }
        const indices = new Uint16Array(24);
        for (let i = 0; i < 24; i++)
            indices[i] = i;
        return createGeometry(gl, program, positions, normals, indices, gl.TRIANGLES);
    }
    // ─── Cylinder ────────────────────────────────────────────────────────
    function createCylinder(gl, program, radius, height, segments) {
        const positions = [];
        const normals = [];
        const indices = [];
        const halfH = height / 2;
        // Side vertices
        for (let i = 0; i <= segments; i++) {
            const theta = (i / segments) * Math.PI * 2;
            const cos = Math.cos(theta);
            const sin = Math.sin(theta);
            const x = radius * cos;
            const z = radius * sin;
            // Top vertex
            positions.push(x, halfH, z);
            normals.push(cos, 0, sin);
            // Bottom vertex
            positions.push(x, -halfH, z);
            normals.push(cos, 0, sin);
        }
        // Side indices
        for (let i = 0; i < segments; i++) {
            const a = i * 2;
            const b = a + 1;
            const c = a + 2;
            const d = a + 3;
            indices.push(a, b, c, b, d, c);
        }
        // Top cap center
        const topCenter = positions.length / 3;
        positions.push(0, halfH, 0);
        normals.push(0, 1, 0);
        for (let i = 0; i <= segments; i++) {
            const theta = (i / segments) * Math.PI * 2;
            positions.push(radius * Math.cos(theta), halfH, radius * Math.sin(theta));
            normals.push(0, 1, 0);
        }
        for (let i = 0; i < segments; i++) {
            indices.push(topCenter, topCenter + 1 + i, topCenter + 2 + i);
        }
        // Bottom cap center
        const botCenter = positions.length / 3;
        positions.push(0, -halfH, 0);
        normals.push(0, -1, 0);
        for (let i = 0; i <= segments; i++) {
            const theta = (i / segments) * Math.PI * 2;
            positions.push(radius * Math.cos(theta), -halfH, radius * Math.sin(theta));
            normals.push(0, -1, 0);
        }
        for (let i = 0; i < segments; i++) {
            indices.push(botCenter, botCenter + 2 + i, botCenter + 1 + i);
        }
        return createGeometry(gl, program, new Float32Array(positions), new Float32Array(normals), new Uint16Array(indices), gl.TRIANGLES);
    }
    // ─── Plane (unit quad in XY) ─────────────────────────────────────────
    function createPlane(gl, program) {
        const positions = new Float32Array([
            -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
        ]);
        const normals = new Float32Array([
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        ]);
        const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
        return createGeometry(gl, program, positions, normals, indices, gl.TRIANGLES);
    }
    // ─── Shared geometry creation helper ─────────────────────────────────
    function createGeometry(gl, program, positions, normals, indices, drawMode) {
        const vao = gl.createVertexArray();
        if (!vao)
            throw new Error('[transformer-viz] Failed to create VAO.');
        gl.bindVertexArray(vao);
        // Position buffer
        const posBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        const posLoc = getAttribLocation(gl, program, 'aPosition');
        if (posLoc >= 0) {
            gl.enableVertexAttribArray(posLoc);
            gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
        }
        // Normal buffer
        const normBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, normBuf);
        gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
        const normLoc = getAttribLocation(gl, program, 'aNormal');
        if (normLoc >= 0) {
            gl.enableVertexAttribArray(normLoc);
            gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);
        }
        // Index buffer
        const idxBuf = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
        gl.bindVertexArray(null);
        return { vao, indexCount: indices.length, drawMode };
    }
    /**
     * Create a simple line geometry from a list of 3D points.
     * Used for connection arrows between components.
     */
    function createLineGeometry(gl, program, points) {
        const vao = gl.createVertexArray();
        if (!vao)
            throw new Error('[transformer-viz] Failed to create VAO.');
        gl.bindVertexArray(vao);
        const posBuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
        gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);
        const posLoc = getAttribLocation(gl, program, 'aPosition');
        if (posLoc >= 0) {
            gl.enableVertexAttribArray(posLoc);
            gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
        }
        gl.bindVertexArray(null);
        return { vao, indexCount: points.length / 3, drawMode: gl.LINE_STRIP };
    }

    /**
     * Main render pass: traverses the scene graph and issues draw calls.
     */
    // Pre-allocated temporary matrices
    const _modelView = mat4Create();
    const _normalMat = new Float32Array(9);
    const _mvp$1 = mat4Create();
    /**
     * Extract the 3x3 normal matrix from a 4x4 modelView matrix.
     * This is the inverse transpose of the upper-left 3x3.
     * For uniform scale, we can skip the inverse transpose and just use the 3x3 directly.
     */
    function extractNormalMatrix(out, modelView) {
        // For simplicity (and our use case with uniform-ish scales), use upper-left 3x3 directly.
        // This is correct when there is no non-uniform scaling.
        out[0] = modelView[0];
        out[1] = modelView[1];
        out[2] = modelView[2];
        out[3] = modelView[4];
        out[4] = modelView[5];
        out[5] = modelView[6];
        out[6] = modelView[8];
        out[7] = modelView[9];
        out[8] = modelView[10];
    }
    /**
     * Render the scene graph using the solid shader program.
     */
    function renderScene(ctx, camera, root, connections) {
        const { gl } = ctx;
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        const view = camera.viewMatrix;
        const proj = camera.projMatrix;
        // ─── Render solid components ───────────────────────────────────
        const solidProg = ctx.programs.get(PROGRAM.SOLID);
        const gradientProg = ctx.programs.get(PROGRAM.GRADIENT);
        traverse(root, (node) => {
            if (!node.renderable)
                return;
            const { visualState } = node;
            const hasGradient = visualState.activationMagnitude > 0.01 || visualState.gradientMagnitude > 0.01;
            const prog = hasGradient ? gradientProg : solidProg;
            gl.useProgram(prog);
            // Compute modelView = view * world
            mat4Multiply(_modelView, view, node.worldMatrix);
            // Set uniforms
            const modelViewLoc = getUniformLocation(gl, prog, 'uModelView');
            const projLoc = getUniformLocation(gl, prog, 'uProjection');
            const normalMatLoc = getUniformLocation(gl, prog, 'uNormalMatrix');
            const opacityLoc = getUniformLocation(gl, prog, 'uOpacity');
            if (modelViewLoc)
                gl.uniformMatrix4fv(modelViewLoc, false, _modelView);
            if (projLoc)
                gl.uniformMatrix4fv(projLoc, false, proj);
            extractNormalMatrix(_normalMat, _modelView);
            if (normalMatLoc)
                gl.uniformMatrix3fv(normalMatLoc, false, _normalMat);
            const opacity = visualState.opacity * (visualState.hovered ? 1.0 : 0.92);
            if (opacityLoc)
                gl.uniform1f(opacityLoc, opacity);
            if (hasGradient) {
                // Gradient-mapped program uniforms
                const baseColorLoc = getUniformLocation(gl, prog, 'uBaseColor');
                const gradColorLoc = getUniformLocation(gl, prog, 'uGradientColor');
                const gradMixLoc = getUniformLocation(gl, prog, 'uGradientMix');
                if (baseColorLoc)
                    gl.uniform4fv(baseColorLoc, visualState.baseColor);
                // Activation = yellow glow, Gradient = red-orange
                const totalMix = Math.min(1, visualState.activationMagnitude + visualState.gradientMagnitude);
                const gradColor = visualState.gradientMagnitude > visualState.activationMagnitude
                    ? [0.9, 0.3, 0.1, 1.0] // red-orange for gradients
                    : [1.0, 0.9, 0.3, 1.0]; // yellow for activations
                if (gradColorLoc)
                    gl.uniform4fv(gradColorLoc, gradColor);
                if (gradMixLoc)
                    gl.uniform1f(gradMixLoc, totalMix * 0.6); // Don't fully overwrite base color
            }
            else {
                // Solid program uniform
                const colorLoc = getUniformLocation(gl, prog, 'uColor');
                const c = visualState.baseColor;
                // Brighten on hover/select
                if (visualState.selected) {
                    if (colorLoc)
                        gl.uniform4f(colorLoc, Math.min(1, c[0] + 0.15), Math.min(1, c[1] + 0.15), Math.min(1, c[2] + 0.15), c[3]);
                }
                else if (visualState.hovered) {
                    if (colorLoc)
                        gl.uniform4f(colorLoc, Math.min(1, c[0] + 0.08), Math.min(1, c[1] + 0.08), Math.min(1, c[2] + 0.08), c[3]);
                }
                else {
                    if (colorLoc)
                        gl.uniform4fv(colorLoc, c);
                }
            }
            // Draw
            gl.bindVertexArray(node.renderable.vao);
            gl.drawElements(gl.TRIANGLES, node.renderable.indexCount, gl.UNSIGNED_SHORT, 0);
            gl.bindVertexArray(null);
        });
        // ─── Render connection lines ───────────────────────────────────
        if (connections.length > 0) {
            const lineProg = ctx.programs.get(PROGRAM.LINE);
            gl.useProgram(lineProg);
            mat4Multiply(_mvp$1, proj, view); // No model transform for lines (they're in world space)
            const mvpLoc = getUniformLocation(gl, lineProg, 'uMVP');
            const colorLoc = getUniformLocation(gl, lineProg, 'uColor');
            if (mvpLoc)
                gl.uniformMatrix4fv(mvpLoc, false, _mvp$1);
            gl.lineWidth(1.0); // WebGL only supports 1.0 on most implementations
            for (const conn of connections) {
                if (colorLoc)
                    gl.uniform4fv(colorLoc, conn.color);
                gl.bindVertexArray(conn.geometry.vao);
                gl.drawArrays(gl.LINE_STRIP, 0, conn.geometry.indexCount);
                gl.bindVertexArray(null);
            }
        }
    }

    /**
     * Color-based picking: render scene to offscreen FBO with unique colors per pickable node.
     */
    const _mvp = mat4Create();
    const _tempViewProj = mat4Create();
    /**
     * Create the offscreen framebuffer for picking.
     * Uses half resolution for performance.
     */
    function createPickTarget(gl, width, height) {
        // Half resolution
        const w = Math.max(1, Math.floor(width / 2));
        const h = Math.max(1, Math.floor(height / 2));
        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        const colorTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, colorTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTex, 0);
        const depthRb = gl.createRenderbuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, depthRb);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRb);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return { fbo, colorTex, depthRb, width: w, height: h };
    }
    /**
     * Render the pick pass to the offscreen FBO.
     */
    function renderPickPass(ctx, camera, root, target) {
        const { gl } = ctx;
        const prog = ctx.programs.get(PROGRAM.PICKING);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
        gl.viewport(0, 0, target.width, target.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.disable(gl.BLEND); // No blending for picking
        gl.useProgram(prog);
        mat4Multiply(_tempViewProj, camera.projMatrix, camera.viewMatrix);
        const mvpLoc = getUniformLocation(gl, prog, 'uMVP');
        const pickColorLoc = getUniformLocation(gl, prog, 'uPickColor');
        traverse(root, (node) => {
            if (!node.renderable || node.pickId === 0)
                return;
            mat4Multiply(_mvp, _tempViewProj, node.worldMatrix);
            if (mvpLoc)
                gl.uniformMatrix4fv(mvpLoc, false, _mvp);
            // Encode pickId into RGB (24-bit). Alpha = 1.
            const r = ((node.pickId >> 16) & 0xFF) / 255;
            const g = ((node.pickId >> 8) & 0xFF) / 255;
            const b = (node.pickId & 0xFF) / 255;
            if (pickColorLoc)
                gl.uniform4f(pickColorLoc, r, g, b, 1.0);
            gl.bindVertexArray(node.renderable.vao);
            gl.drawElements(gl.TRIANGLES, node.renderable.indexCount, gl.UNSIGNED_SHORT, 0);
            gl.bindVertexArray(null);
        });
        // Restore state
        gl.enable(gl.BLEND);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        // Restore main viewport
        gl.viewport(0, 0, ctx.canvas.width, ctx.canvas.height);
        gl.clearColor(0.96, 0.96, 0.96, 1.0);
    }
    /**
     * Read the pick ID at screen coordinates (x, y).
     * Returns the SceneNode if found, null otherwise.
     */
    function pickAtScreen(gl, target, root, screenX, screenY, canvasWidth, canvasHeight) {
        // Convert screen coords to pick FBO coords (half resolution, Y flipped)
        const pickX = Math.floor(screenX * (target.width / canvasWidth));
        const pickY = target.height - 1 - Math.floor(screenY * (target.height / canvasHeight));
        if (pickX < 0 || pickX >= target.width || pickY < 0 || pickY >= target.height) {
            return null;
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
        const pixel = new Uint8Array(4);
        gl.readPixels(pickX, pickY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        const pickId = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];
        if (pickId === 0)
            return null;
        return findByPickId(root, pickId);
    }
    /**
     * Destroy pick target GL resources.
     */
    function destroyPickTarget(gl, target) {
        gl.deleteFramebuffer(target.fbo);
        gl.deleteTexture(target.colorTex);
        gl.deleteRenderbuffer(target.depthRb);
    }

    /**
     * Canvas 2D overlay for rendering text labels projected from 3D positions.
     */
    /**
     * Create a text overlay canvas positioned on top of the WebGL canvas.
     */
    function createTextOverlay(container, width, height) {
        const canvas = document.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.pointerEvents = 'none'; // Click-through
        container.appendChild(canvas);
        const ctx = canvas.getContext('2d');
        let dpr = window.devicePixelRatio || 1;
        function resize(w, h) {
            dpr = window.devicePixelRatio || 1;
            canvas.width = Math.round(w * dpr);
            canvas.height = Math.round(h * dpr);
            canvas.style.width = `${w}px`;
            canvas.style.height = `${h}px`;
            ctx.scale(dpr, dpr);
        }
        resize(width, height);
        const _labelPos = vec3Create(0, 0, 0);
        function render(camera, root) {
            const w = canvas.width / dpr;
            const h = canvas.height / dpr;
            // Clear
            ctx.save();
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);
            const viewProj = camera.viewProjMatrix;
            // Font settings
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            traverse(root, (node) => {
                if (!node.label || !node.visible)
                    return;
                // Don't show labels for expanded group nodes — their children have labels
                if (node.expanded)
                    return;
                // Get world position of this node's center
                _labelPos[0] = node.worldMatrix[12];
                _labelPos[1] = node.worldMatrix[13];
                _labelPos[2] = node.worldMatrix[14];
                // Offset label above the node's top edge
                // Use the actual rendered height (scale[1]) plus a small gap
                const halfHeight = node.renderable ? node.scale[1] * 0.5 : 0;
                _labelPos[1] += halfHeight + 0.25;
                const screen = projectToScreen(_labelPos, viewProj, w, h);
                if (!screen)
                    return; // Behind camera
                if (screen.x < -50 || screen.x > w + 50 || screen.y < -50 || screen.y > h + 50)
                    return; // Off screen
                // Compute font size based on distance (closer = larger)
                const distFactor = Math.max(0.5, Math.min(1.5, 20 / Math.max(1, camera.state.distance)));
                const fontSize = Math.round(12 * distFactor);
                ctx.font = `${fontSize}px -apple-system, system-ui, sans-serif`;
                // Background pill
                const textWidth = ctx.measureText(node.label).width;
                const padding = 4;
                const pillW = textWidth + padding * 2;
                const pillH = fontSize + padding;
                ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
                ctx.beginPath();
                roundRect(ctx, screen.x - pillW / 2, screen.y - pillH / 2, pillW, pillH, 3);
                ctx.fill();
                // Border
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
                // Text
                ctx.fillStyle = node.visualState.selected ? '#2563eb' : '#1a1a1a';
                ctx.fillText(node.label, screen.x, screen.y);
            });
            ctx.restore();
        }
        return {
            canvas,
            ctx,
            resize,
            render,
            destroy() {
                canvas.remove();
            },
        };
    }
    /** Polyfill-safe rounded rectangle path. */
    function roundRect(ctx, x, y, w, h, r) {
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    /**
     * Orbit camera controls via pointer events.
     * Left-drag: orbit. Scroll: zoom. Shift+drag or middle-drag: pan. Double-click: reset.
     */
    /**
     * Attach orbit controls to a canvas element.
     */
    function createOrbitControls(canvas, camera, onUpdate, config) {
        const orbitSens = 0.005;
        const zoomSens = 0.05;
        let isDragging = false;
        let isPanning = false;
        let lastX = 0;
        let lastY = 0;
        let pointerId = null;
        function onPointerDown(e) {
            if (pointerId !== null)
                return; // Only track one pointer
            pointerId = e.pointerId;
            isDragging = true;
            isPanning = e.shiftKey || e.button === 1; // Shift+left or middle button
            lastX = e.clientX;
            lastY = e.clientY;
            canvas.setPointerCapture(e.pointerId);
            e.preventDefault();
        }
        function onPointerMove(e) {
            if (!isDragging || e.pointerId !== pointerId)
                return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            if (isPanning) {
                camera.pan(-dx, dy);
            }
            else {
                camera.orbit(-dx * orbitSens, -dy * orbitSens);
            }
            onUpdate();
        }
        function onPointerUp(e) {
            if (e.pointerId !== pointerId)
                return;
            isDragging = false;
            isPanning = false;
            pointerId = null;
        }
        function onWheel(e) {
            e.preventDefault();
            camera.zoom(e.deltaY * zoomSens);
            onUpdate();
        }
        function onDblClick(_e) {
            camera.reset();
            onUpdate();
        }
        // Prevent context menu on right-click
        function onContextMenu(e) {
            e.preventDefault();
        }
        canvas.addEventListener('pointerdown', onPointerDown);
        canvas.addEventListener('pointermove', onPointerMove);
        canvas.addEventListener('pointerup', onPointerUp);
        canvas.addEventListener('pointercancel', onPointerUp);
        canvas.addEventListener('wheel', onWheel, { passive: false });
        canvas.addEventListener('dblclick', onDblClick);
        canvas.addEventListener('contextmenu', onContextMenu);
        return {
            destroy() {
                canvas.removeEventListener('pointerdown', onPointerDown);
                canvas.removeEventListener('pointermove', onPointerMove);
                canvas.removeEventListener('pointerup', onPointerUp);
                canvas.removeEventListener('pointercancel', onPointerUp);
                canvas.removeEventListener('wheel', onWheel);
                canvas.removeEventListener('dblclick', onDblClick);
                canvas.removeEventListener('contextmenu', onContextMenu);
            },
        };
    }

    /**
     * Picking interaction: maps mouse events to scene nodes via the pick FBO.
     */
    /**
     * Set up picking event handlers on the WebGL canvas.
     */
    function createPickingHandler(ctx, camera, target, root, callbacks) {
        const { gl, canvas } = ctx;
        let _root = root;
        let _hoveredNode = null;
        function getCanvasCoords(e) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
            };
        }
        function onPointerMove(e) {
            // Don't pick during drag
            if (e.buttons !== 0)
                return;
            const { x, y } = getCanvasCoords(e);
            const logicalWidth = parseInt(canvas.style.width);
            const logicalHeight = parseInt(canvas.style.height);
            const node = pickAtScreen(gl, target, _root, x, y, logicalWidth, logicalHeight);
            if (node !== _hoveredNode) {
                // Clear old hover
                if (_hoveredNode) {
                    _hoveredNode.visualState.hovered = false;
                }
                // Set new hover
                if (node) {
                    node.visualState.hovered = true;
                    canvas.style.cursor = 'pointer';
                }
                else {
                    canvas.style.cursor = 'default';
                }
                _hoveredNode = node;
                callbacks.onHover(node);
            }
        }
        function onClick(e) {
            const { x, y } = getCanvasCoords(e);
            const logicalWidth = parseInt(canvas.style.width);
            const logicalHeight = parseInt(canvas.style.height);
            const node = pickAtScreen(gl, target, _root, x, y, logicalWidth, logicalHeight);
            callbacks.onClick(node);
        }
        canvas.addEventListener('pointermove', onPointerMove);
        canvas.addEventListener('click', onClick);
        return {
            updatePickBuffer(newRoot) {
                _root = newRoot;
                renderPickPass(ctx, camera, _root, target);
            },
            destroy() {
                canvas.removeEventListener('pointermove', onPointerMove);
                canvas.removeEventListener('click', onClick);
            },
        };
    }

    /**
     * Inspector panel: shows details for a selected component.
     * Rendered as a DOM overlay to the right of the visualization.
     */
    /**
     * Create the inspector panel inside the given container.
     */
    function createInspector(container) {
        const panel = document.createElement('div');
        panel.style.cssText = `
    position: absolute;
    top: 12px;
    right: 12px;
    width: 280px;
    max-height: calc(100% - 24px);
    overflow-y: auto;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    font-family: -apple-system, system-ui, sans-serif;
    font-size: 13px;
    color: #1a1a1a;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    display: none;
    z-index: 10;
  `;
        container.appendChild(panel);
        let _visible = false;
        let _liveDataEl = null;
        let _weightDataEl = null;
        function formatNumber(n) {
            if (n >= 1e6)
                return (n / 1e6).toFixed(1) + 'M';
            if (n >= 1e3)
                return (n / 1e3).toFixed(1) + 'K';
            return n.toLocaleString();
        }
        function formatFloat(n, digits = 4) {
            if (Math.abs(n) < 1e-4 && n !== 0)
                return n.toExponential(2);
            return n.toFixed(digits);
        }
        function show(node) {
            const spec = node.componentSpec;
            if (!spec) {
                hide();
                return;
            }
            let html = `
      <div style="margin-bottom: 12px;">
        <div style="font-size: 15px; font-weight: 600; margin-bottom: 4px;">${esc(spec.label)}</div>
        <div style="color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">${esc(spec.type)}</div>
      </div>
    `;
            const d = spec.details;
            if (d.paramCount !== undefined) {
                html += `<div style="margin-bottom: 8px;"><span style="color: #888;">Parameters:</span> ${formatNumber(d.paramCount)}</div>`;
            }
            if (d.inputShape) {
                html += `<div style="margin-bottom: 4px;"><span style="color: #888;">Input:</span> <code style="background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px;">${esc(d.inputShape)}</code></div>`;
            }
            if (d.outputShape) {
                html += `<div style="margin-bottom: 8px;"><span style="color: #888;">Output:</span> <code style="background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px;">${esc(d.outputShape)}</code></div>`;
            }
            // Weights shape table (from spec — always shown if declared)
            if (d.weights && Object.keys(d.weights).length > 0) {
                html += `<div style="margin: 12px 0 6px; font-weight: 500; font-size: 12px; color: #555;">Weights</div>`;
                html += `<table style="width: 100%; border-collapse: collapse; font-size: 12px;">`;
                for (const [name, shape] of Object.entries(d.weights)) {
                    html += `<tr>
          <td style="padding: 2px 0; color: #333; font-family: monospace; font-size: 11px;">${esc(name)}</td>
          <td style="padding: 2px 0; text-align: right; color: #666;">[${shape.join(', ')}]</td>
        </tr>`;
                }
                html += `</table>`;
            }
            // Config
            if (d.config && Object.keys(d.config).length > 0) {
                html += `<div style="margin: 12px 0 6px; font-weight: 500; font-size: 12px; color: #555;">Config</div>`;
                for (const [key, val] of Object.entries(d.config)) {
                    html += `<div style="font-size: 12px; margin-bottom: 2px;"><span style="color: #888;">${esc(key)}:</span> ${esc(String(val))}</div>`;
                }
            }
            // Weight statistics placeholder (populated by updateWeightData)
            html += `<div id="inspector-weight-data" style="display: none;"></div>`;
            // Live data placeholder
            html += `<div id="inspector-live-data" style="margin-top: 12px; display: none;">
      <div style="border-top: 1px solid #e0e0e0; padding-top: 8px; margin-top: 4px; font-weight: 500; font-size: 12px; color: #555;">Live Data</div>
      <div id="inspector-live-values" style="font-size: 12px; margin-top: 4px;"></div>
    </div>`;
            // Close button
            html += `<div style="position: absolute; top: 8px; right: 10px; cursor: pointer; color: #999; font-size: 16px; line-height: 1;" id="inspector-close">&times;</div>`;
            panel.innerHTML = html;
            panel.style.display = 'block';
            _visible = true;
            _liveDataEl = panel.querySelector('#inspector-live-data');
            _weightDataEl = panel.querySelector('#inspector-weight-data');
            const closeBtn = panel.querySelector('#inspector-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', hide);
            }
        }
        function hide() {
            panel.style.display = 'none';
            _visible = false;
            _liveDataEl = null;
            _weightDataEl = null;
        }
        function updateLiveData(data) {
            if (!_liveDataEl)
                return;
            const hasData = data.activationNorm !== undefined || data.gradientNorm !== undefined || data.maxAttnWeight !== undefined;
            _liveDataEl.style.display = hasData ? 'block' : 'none';
            if (!hasData)
                return;
            const valuesEl = _liveDataEl.querySelector('#inspector-live-values');
            if (!valuesEl)
                return;
            let html = '';
            if (data.activationNorm !== undefined) {
                html += `<div style="margin-bottom: 2px;"><span style="color: #888;">Activation norm:</span> ${data.activationNorm.toFixed(4)}</div>`;
            }
            if (data.gradientNorm !== undefined) {
                html += `<div style="margin-bottom: 2px;"><span style="color: #888;">Gradient norm:</span> ${data.gradientNorm.toFixed(6)}</div>`;
            }
            if (data.maxAttnWeight !== undefined) {
                html += `<div style="margin-bottom: 2px;"><span style="color: #888;">Max attn weight:</span> ${data.maxAttnWeight.toFixed(4)}</div>`;
            }
            valuesEl.innerHTML = html;
        }
        function updateWeightData(data) {
            if (!_weightDataEl)
                return;
            if (data.weights.length === 0) {
                _weightDataEl.style.display = 'none';
                return;
            }
            _weightDataEl.style.display = 'block';
            let html = `<div style="border-top: 1px solid #e0e0e0; padding-top: 8px; margin-top: 12px;">
      <div style="font-weight: 500; font-size: 12px; color: #555; margin-bottom: 6px;">Weight Statistics</div>
      <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
        Total: ${formatNumber(data.totalParams)} params, RMS norm: ${formatFloat(data.totalNorm)}
      </div>`;
            for (const w of data.weights) {
                html += renderWeightStats(w);
            }
            html += `</div>`;
            _weightDataEl.innerHTML = html;
        }
        function renderWeightStats(w) {
            let html = `<div style="margin-bottom: 10px; padding: 8px; background: #f8f8f6; border-radius: 6px; border: 1px solid #eee;">`;
            // Header
            html += `<div style="font-family: monospace; font-size: 11px; font-weight: 600; color: #333; margin-bottom: 4px;">${esc(w.name)}</div>`;
            html += `<div style="font-size: 11px; color: #888; margin-bottom: 6px;">[${w.shape.join(' x ')}] = ${formatNumber(w.numParams)} params</div>`;
            // Stats grid
            html += `<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2px 12px; font-size: 11px; margin-bottom: 6px;">`;
            html += `<div><span style="color: #888;">mean:</span> ${formatFloat(w.mean)}</div>`;
            html += `<div><span style="color: #888;">std:</span> ${formatFloat(w.std)}</div>`;
            html += `<div><span style="color: #888;">min:</span> ${formatFloat(w.min)}</div>`;
            html += `<div><span style="color: #888;">max:</span> ${formatFloat(w.max)}</div>`;
            html += `<div><span style="color: #888;">L2 norm:</span> ${formatFloat(w.l2Norm)}</div>`;
            html += `<div><span style="color: #888;">sparsity:</span> ${(w.sparsity * 100).toFixed(1)}%</div>`;
            html += `</div>`;
            // Mini histogram (inline SVG)
            html += renderHistogram(w.histogram);
            html += `</div>`;
            return html;
        }
        function renderHistogram(hist) {
            const width = 240;
            const height = 40;
            const maxCount = Math.max(...hist.counts);
            if (maxCount === 0)
                return '';
            const barWidth = width / hist.counts.length;
            let bars = '';
            for (let i = 0; i < hist.counts.length; i++) {
                const barHeight = (hist.counts[i] / maxCount) * height;
                const x = i * barWidth;
                const y = height - barHeight;
                // Color: center bins (near zero) lighter, tails darker
                const t = i / hist.counts.length;
                const distFromCenter = Math.abs(t - 0.5) * 2; // 0 at center, 1 at edges
                const r = Math.round(37 + distFromCenter * 60);
                const g = Math.round(99 - distFromCenter * 40);
                const b = Math.round(235 - distFromCenter * 80);
                bars += `<rect x="${x}" y="${y}" width="${barWidth - 0.5}" height="${barHeight}" fill="rgb(${r},${g},${b})" rx="1"/>`;
            }
            // Axis labels
            const minLabel = hist.min.toFixed(2);
            const maxLabel = hist.max.toFixed(2);
            return `<svg viewBox="0 0 ${width} ${height + 14}" width="${width}" height="${height + 14}" style="display: block;">
      ${bars}
      <text x="0" y="${height + 11}" font-size="9" fill="#999" font-family="-apple-system, system-ui, sans-serif">${minLabel}</text>
      <text x="${width}" y="${height + 11}" font-size="9" fill="#999" text-anchor="end" font-family="-apple-system, system-ui, sans-serif">${maxLabel}</text>
    </svg>`;
        }
        return {
            show,
            hide,
            updateLiveData,
            updateWeightData,
            destroy() {
                panel.remove();
            },
            get isVisible() { return _visible; },
        };
    }
    function esc(s) {
        return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    /**
     * Layout constants for positioning components in 3D space.
     */
    const LAYOUT = {
        /** Vertical space per transformer layer (collapsed). */
        LAYER_HEIGHT: 3.0,
        /** Vertical space per transformer layer (expanded). */
        LAYER_HEIGHT_EXPANDED: 8.0,
        /** Width of a standard component box. */
        BLOCK_WIDTH: 2.0,
        /** Depth (Z) of a standard component box. */
        BLOCK_DEPTH: 0.8,
        /** Height of a standard component box. */
        BLOCK_HEIGHT: 0.7,
        /** Center-to-center horizontal distance between attention and MLP in parallel layout.
         *  Dynamically increased when attention is expanded. */
        PARALLEL_GAP: 3.0,
        /** Extra parallel gap added when attention is expanded (heads need more room). */
        PARALLEL_GAP_EXPANDED: 5.5,
        /** Vertical gap between sequential components. */
        VERTICAL_GAP: 0.5,
        /** Height for the embedding block (taller). */
        EMBEDDING_HEIGHT: 1.0,
        /** Height for layernorm blocks (thin). */
        LAYERNORM_HEIGHT: 0.25,
        /** Height for the output head block. */
        OUTPUT_HEAD_HEIGHT: 0.7,
        /** Size of diamond (residual add) blocks. */
        DIAMOND_SIZE: 0.4,
        /** Extra margin beyond the scene's rightmost edge for residual/skip connections. */
        RESIDUAL_SPINE_OFFSET: 0.8,
        /** Gap between component surface and connection arrow. */
        ARROW_GAP: 0.1,
        // ─── Attention internals (expanded view) ───────────────────────
        /** Width of the QKV / output projection box. */
        ATTN_PROJ_WIDTH: 1.6,
        /** Height of the QKV / output projection box. */
        ATTN_PROJ_HEIGHT: 0.35,
        /** Width of a single attention head box. */
        ATTN_HEAD_WIDTH: 0.5,
        /** Height of a single attention head box. */
        ATTN_HEAD_HEIGHT: 0.45,
        /** Horizontal gap between attention head boxes. */
        ATTN_HEAD_GAP: 0.15,
        /** Vertical gap within attention internals. */
        ATTN_INTERNAL_GAP: 0.35,
        /** Height of the output projection box. */
        ATTN_OUT_HEIGHT: 0.35,
        /** Total expanded attention height (computed dynamically, this is a minimum). */
        ATTN_EXPANDED_HEIGHT: 5.0,
    };
    /**
     * Color palette for components (RGBA, values 0-1).
     */
    const COMPONENT_COLORS = {
        embedding: [0.31, 0.27, 0.90, 1.0], // Indigo #4f46e5
        layernorm: [0.42, 0.45, 0.50, 1.0], // Gray #6b7280
        attention: [0.15, 0.39, 0.92, 1.0], // Blue #2563eb
        mlp: [0.09, 0.64, 0.29, 1.0], // Green #16a34a
        transformer_layer: [0.80, 0.80, 0.80, 0.15], // Transparent gray
        output_head: [0.86, 0.15, 0.15, 1.0], // Red #dc2626
        residual_add: [0.96, 0.62, 0.04, 1.0], // Amber #f59e0b
        linear: [0.50, 0.50, 0.60, 1.0], // Slate
        custom: [0.60, 0.60, 0.60, 1.0], // Gray
        // Attention internals
        attn_qkv_proj: [0.20, 0.45, 0.85, 1.0], // Medium blue
        attn_out_proj: [0.25, 0.35, 0.75, 1.0], // Darker blue
        attn_concat: [0.35, 0.50, 0.80, 1.0], // Blue-gray
    };
    /** Per-head colors for multi-head attention visualization. */
    const HEAD_COLORS = [
        [0.15, 0.50, 0.92, 1.0], // Blue
        [0.60, 0.20, 0.85, 1.0], // Purple
        [0.10, 0.70, 0.55, 1.0], // Teal
        [0.90, 0.50, 0.15, 1.0], // Orange
        [0.85, 0.20, 0.50, 1.0], // Pink
        [0.30, 0.65, 0.20, 1.0], // Green
        [0.70, 0.60, 0.10, 1.0], // Gold
        [0.50, 0.35, 0.70, 1.0], // Lavender
    ];

    /**
     * Maps ComponentSpec.type to SceneNode builders.
     * Each builder creates the visual representation (geometry + transform) for a component.
     */
    /**
     * Build a SceneNode subtree for the given component spec.
     * When `expandAttention` is true and spec is an attention block, builds the multi-head view.
     */
    function buildComponentNode(spec, primitives, expandAttention = false) {
        switch (spec.type) {
            case 'embedding': return buildBox(spec, primitives.box, LAYOUT.BLOCK_WIDTH * 1.2, LAYOUT.EMBEDDING_HEIGHT, LAYOUT.BLOCK_DEPTH);
            case 'layernorm': return buildBox(spec, primitives.box, LAYOUT.BLOCK_WIDTH, LAYOUT.LAYERNORM_HEIGHT, LAYOUT.BLOCK_DEPTH * 0.6);
            case 'attention':
                return expandAttention
                    ? buildExpandedAttention(spec, primitives)
                    : buildBox(spec, primitives.box, LAYOUT.BLOCK_WIDTH, LAYOUT.BLOCK_HEIGHT, LAYOUT.BLOCK_DEPTH);
            case 'mlp': return buildBox(spec, primitives.box, LAYOUT.BLOCK_WIDTH, LAYOUT.BLOCK_HEIGHT, LAYOUT.BLOCK_DEPTH);
            case 'transformer_layer': return buildTransformerLayer(spec, primitives);
            case 'output_head': return buildBox(spec, primitives.box, LAYOUT.BLOCK_WIDTH, LAYOUT.OUTPUT_HEAD_HEIGHT, LAYOUT.BLOCK_DEPTH);
            case 'residual_add': return buildDiamond(spec, primitives.diamond);
            case 'linear': return buildBox(spec, primitives.box, LAYOUT.BLOCK_WIDTH * 0.8, LAYOUT.BLOCK_HEIGHT * 0.7, LAYOUT.BLOCK_DEPTH * 0.6);
            case 'custom': return buildBox(spec, primitives.box, LAYOUT.BLOCK_WIDTH, LAYOUT.BLOCK_HEIGHT, LAYOUT.BLOCK_DEPTH);
            default: return buildBox(spec, primitives.box, LAYOUT.BLOCK_WIDTH, LAYOUT.BLOCK_HEIGHT, LAYOUT.BLOCK_DEPTH);
        }
    }
    // ─── Builders ────────────────────────────────────────────────────────
    function buildBox(spec, boxGeometry, width, height, depth) {
        const node = createSceneNode(spec.id, true);
        node.componentSpec = spec;
        node.label = spec.label;
        setScale(node, width, height, depth);
        const color = COMPONENT_COLORS[spec.type] ?? COMPONENT_COLORS.custom;
        node.visualState.baseColor = [...color];
        node.renderable = {
            programId: 'solid',
            vao: boxGeometry.vao,
            indexCount: boxGeometry.indexCount,
            uniforms: {},
        };
        return node;
    }
    function buildDiamond(spec, diamondGeometry) {
        const node = createSceneNode(spec.id, true);
        node.componentSpec = spec;
        node.label = spec.label;
        const s = LAYOUT.DIAMOND_SIZE;
        setScale(node, s, s, s);
        const color = COMPONENT_COLORS.residual_add;
        node.visualState.baseColor = [...color];
        node.renderable = {
            programId: 'solid',
            vao: diamondGeometry.vao,
            indexCount: diamondGeometry.indexCount,
            uniforms: {},
        };
        return node;
    }
    function buildTransformerLayer(spec, primitives) {
        const node = createSceneNode(spec.id, true);
        node.componentSpec = spec;
        node.label = spec.label;
        node.expanded = false;
        node.visualState.baseColor = [0.45, 0.45, 0.55, 1.0];
        node.visualState.opacity = 1.0;
        setScale(node, LAYOUT.BLOCK_WIDTH * 1.4, LAYOUT.BLOCK_HEIGHT * 1.2, LAYOUT.BLOCK_DEPTH);
        node.renderable = {
            programId: 'solid',
            vao: primitives.box.vao,
            indexCount: primitives.box.indexCount,
            uniforms: {},
        };
        return node;
    }
    // ─── Expanded Attention (multi-head view) ────────────────────────────
    /**
     * Build the expanded attention block showing multi-head internals:
     *
     *   ┌──────────────────┐
     *   │  QKV Projection  │
     *   └──────────────────┘
     *          │
     *    ┌─────┼─────┐
     *   [H0] [H1] [H2] [H3]   ← per-head attention
     *    └─────┼─────┘
     *          │
     *   ┌──────────────────┐
     *   │     Concat       │
     *   └──────────────────┘
     *          │
     *   ┌──────────────────┐
     *   │  Out Projection  │
     *   └──────────────────┘
     */
    function buildExpandedAttention(spec, primitives) {
        const numHeads = spec.details.config?.numHeads ?? 4;
        const headDim = spec.details.config?.headDim ?? 32;
        // Group node — no renderable of its own
        const group = createSceneNode(spec.id, true);
        group.componentSpec = spec;
        group.label = spec.label;
        group.expanded = true;
        setScale(group, 1, 1, 1);
        let localY = 0;
        // 1. QKV Projection box
        const qkvId = `${spec.id}.__qkv_proj`;
        const qkvNode = createSceneNode(qkvId, true);
        qkvNode.componentSpec = {
            id: qkvId,
            type: 'linear',
            label: 'QKV Projection',
            details: {
                inputShape: spec.details.inputShape,
                outputShape: `3 × [seqLen, ${numHeads}, ${headDim}]`,
                paramCount: spec.details.weights?.['qkv.weight']
                    ? spec.details.weights['qkv.weight'].reduce((a, b) => a * b, 1) + (spec.details.weights['qkv.bias']?.[0] ?? 0)
                    : undefined,
                config: { operation: 'Linear → split into Q, K, V' },
            },
        };
        qkvNode.label = 'QKV Proj';
        setScale(qkvNode, LAYOUT.ATTN_PROJ_WIDTH, LAYOUT.ATTN_PROJ_HEIGHT, LAYOUT.BLOCK_DEPTH * 0.7);
        qkvNode.visualState.baseColor = [...COMPONENT_COLORS.attn_qkv_proj];
        qkvNode.renderable = {
            programId: 'solid',
            vao: primitives.box.vao,
            indexCount: primitives.box.indexCount,
            uniforms: {},
        };
        setPosition(qkvNode, 0, localY, 0);
        addChild(group, qkvNode);
        localY -= LAYOUT.ATTN_PROJ_HEIGHT / 2 + LAYOUT.ATTN_INTERNAL_GAP;
        // 2. Per-head attention boxes
        const totalHeadsWidth = numHeads * LAYOUT.ATTN_HEAD_WIDTH + (numHeads - 1) * LAYOUT.ATTN_HEAD_GAP;
        const startX = -totalHeadsWidth / 2 + LAYOUT.ATTN_HEAD_WIDTH / 2;
        localY -= LAYOUT.ATTN_HEAD_HEIGHT / 2;
        for (let h = 0; h < numHeads; h++) {
            const headId = `${spec.id}.__head_${h}`;
            const headNode = createSceneNode(headId, true);
            headNode.componentSpec = {
                id: headId,
                type: 'custom',
                label: `Head ${h}`,
                details: {
                    config: {
                        headIndex: h,
                        headDim,
                        operation: `Q${h}·K${h}ᵀ/√d → softmax → ·V${h}`,
                    },
                    inputShape: `[seqLen, ${headDim}]`,
                    outputShape: `[seqLen, ${headDim}]`,
                },
            };
            headNode.label = `H${h}`;
            setScale(headNode, LAYOUT.ATTN_HEAD_WIDTH, LAYOUT.ATTN_HEAD_HEIGHT, LAYOUT.BLOCK_DEPTH * 0.6);
            const headColor = HEAD_COLORS[h % HEAD_COLORS.length];
            headNode.visualState.baseColor = [...headColor];
            headNode.renderable = {
                programId: 'solid',
                vao: primitives.box.vao,
                indexCount: primitives.box.indexCount,
                uniforms: {},
            };
            const hx = startX + h * (LAYOUT.ATTN_HEAD_WIDTH + LAYOUT.ATTN_HEAD_GAP);
            setPosition(headNode, hx, localY, 0);
            addChild(group, headNode);
        }
        localY -= LAYOUT.ATTN_HEAD_HEIGHT / 2 + LAYOUT.ATTN_INTERNAL_GAP;
        // 3. Concat box
        const concatId = `${spec.id}.__concat`;
        const concatNode = createSceneNode(concatId, true);
        concatNode.componentSpec = {
            id: concatId,
            type: 'custom',
            label: 'Concat Heads',
            details: {
                config: { operation: `Concatenate ${numHeads} heads → [seqLen, ${numHeads * headDim}]` },
            },
        };
        concatNode.label = 'Concat';
        setScale(concatNode, LAYOUT.ATTN_PROJ_WIDTH * 0.9, LAYOUT.ATTN_PROJ_HEIGHT * 0.7, LAYOUT.BLOCK_DEPTH * 0.5);
        concatNode.visualState.baseColor = [...COMPONENT_COLORS.attn_concat];
        concatNode.renderable = {
            programId: 'solid',
            vao: primitives.box.vao,
            indexCount: primitives.box.indexCount,
            uniforms: {},
        };
        setPosition(concatNode, 0, localY, 0);
        addChild(group, concatNode);
        localY -= LAYOUT.ATTN_PROJ_HEIGHT * 0.7 / 2 + LAYOUT.ATTN_INTERNAL_GAP;
        // 4. Output Projection box
        const outId = `${spec.id}.__out_proj`;
        const outNode = createSceneNode(outId, true);
        outNode.componentSpec = {
            id: outId,
            type: 'linear',
            label: 'Output Projection',
            details: {
                outputShape: spec.details.outputShape,
                paramCount: spec.details.weights?.['out.weight']
                    ? spec.details.weights['out.weight'].reduce((a, b) => a * b, 1) + (spec.details.weights['out.bias']?.[0] ?? 0)
                    : undefined,
                config: { operation: 'Linear projection back to model dim' },
            },
        };
        outNode.label = 'Out Proj';
        setScale(outNode, LAYOUT.ATTN_PROJ_WIDTH, LAYOUT.ATTN_OUT_HEIGHT, LAYOUT.BLOCK_DEPTH * 0.7);
        outNode.visualState.baseColor = [...COMPONENT_COLORS.attn_out_proj];
        outNode.renderable = {
            programId: 'solid',
            vao: primitives.box.vao,
            indexCount: primitives.box.indexCount,
            uniforms: {},
        };
        setPosition(outNode, 0, localY, 0);
        addChild(group, outNode);
        // Center all children around Y=0 so the layout engine's rowHeight/2 math works.
        // Children currently span from Y=0 (top) to Y=localY (bottom).
        // Shift everything up by half the total extent.
        const totalExtent = -localY; // positive value
        const offset = totalExtent / 2;
        for (const child of group.children) {
            child.position[1] += offset;
        }
        return group;
    }
    /**
     * Compute the total height of an expanded attention block.
     */
    function getExpandedAttentionHeight(numHeads) {
        return (LAYOUT.ATTN_PROJ_HEIGHT + // QKV proj
            LAYOUT.ATTN_INTERNAL_GAP +
            LAYOUT.ATTN_HEAD_HEIGHT + // heads row
            LAYOUT.ATTN_INTERNAL_GAP +
            LAYOUT.ATTN_PROJ_HEIGHT * 0.7 + // concat
            LAYOUT.ATTN_INTERNAL_GAP +
            LAYOUT.ATTN_OUT_HEIGHT // out proj
        );
    }

    /**
     * Layout engine: reads a ModelSpec and computes 3D positions for all components.
     * Pipeline flows top-to-bottom (Y decreases).
     */
    /**
     * Build the complete scene graph from a ModelSpec.
     */
    function layoutScene(spec, gl, lineProgram, primitives, expandedLayers, expandedAttention, gapOverride) {
        const root = createSceneNode('__root__');
        const nodeMap = new Map();
        const connections = [];
        // Build a lookup of specs by ID
        const specMap = new Map();
        for (const comp of spec.components) {
            specMap.set(comp.id, comp);
        }
        // Identify transformer layers and their children
        const layerChildren = new Set();
        for (const comp of spec.components) {
            if (comp.type === 'transformer_layer' && comp.children) {
                for (const childId of comp.children) {
                    layerChildren.add(childId);
                }
            }
        }
        // Compute the top-level pipeline order (components not inside a layer, plus layers themselves)
        const topLevel = spec.components.filter(c => !layerChildren.has(c.id));
        // Assign Y positions — pipeline flows bottom to top (Y increases upward).
        // Embedding at the bottom (lowest Y), LM Head at the top (highest Y).
        //
        // `cursor` tracks the next available Y — the bottom edge of the next component
        // goes here. This guarantees no overlap regardless of component sizes.
        let cursor = 0;
        const GAP = gapOverride ?? LAYOUT.VERTICAL_GAP;
        for (const comp of topLevel) {
            const isExpanded = comp.type === 'transformer_layer' && expandedLayers?.has(comp.id);
            if (comp.type === 'transformer_layer') {
                if (isExpanded) {
                    const layerNode = buildComponentNode(comp, primitives);
                    layerNode.expanded = true;
                    layerNode.renderable = null;
                    layerNode.label = undefined;
                    const isParallel = comp.details.config?.parallel === true;
                    const children = (comp.children ?? []).map(id => specMap.get(id)).filter(Boolean);
                    const { items, totalHeight } = layoutLayerChildren(children, isParallel, primitives, expandedAttention);
                    // Group center so children span [cursor, cursor + totalHeight]
                    const centerY = cursor + totalHeight / 2;
                    setPosition(layerNode, 0, centerY, 0);
                    setScale(layerNode, 1, 1, 1);
                    // Flip children upward and center around group origin
                    const halfH = totalHeight / 2;
                    for (const { spec: childSpec, node: childNode, y, x } of items) {
                        setPosition(childNode, x, -y - halfH, 0);
                        addChild(layerNode, childNode);
                        nodeMap.set(childSpec.id, childNode);
                    }
                    addChild(root, layerNode);
                    nodeMap.set(comp.id, layerNode);
                    cursor += totalHeight + GAP;
                }
                else {
                    const h = LAYOUT.BLOCK_HEIGHT * 1.2; // collapsed layer box height
                    const centerY = cursor + h / 2;
                    const node = buildComponentNode(comp, primitives);
                    setPosition(node, 0, centerY, 0);
                    addChild(root, node);
                    nodeMap.set(comp.id, node);
                    cursor += h + GAP;
                }
            }
            else {
                const h = getComponentHeight(comp);
                const centerY = cursor + h / 2;
                const node = buildComponentNode(comp, primitives);
                setPosition(node, 0, centerY, 0);
                addChild(root, node);
                nodeMap.set(comp.id, node);
                cursor += h + GAP;
            }
        }
        // Compute the max X extent across all visible nodes (for routing skip connections outside)
        let sceneMaxX = 0;
        for (const node of nodeMap.values()) {
            const wp = getWorldPosition(node);
            const halfW = node.expanded ? 0 : node.scale[0] * 0.5;
            const rightEdge = Math.abs(wp[0]) + halfW;
            if (rightEdge > sceneMaxX)
                sceneMaxX = rightEdge;
        }
        // Build connection line geometries with smart filtering
        for (const conn of spec.connections) {
            let fromId = conn.from;
            let toId = conn.to;
            // --- Filtering rules ---
            // 1. Skip connections whose source is inside a collapsed layer
            if (layerChildren.has(fromId) && !expandedLayers?.has(getParentLayerId(fromId, spec)))
                continue;
            // 2. Skip connections whose target is inside a collapsed layer
            if (layerChildren.has(toId) && !expandedLayers?.has(getParentLayerId(toId, spec)))
                continue;
            // 3. If the target is a transformer_layer that IS expanded, skip the top-level
            //    connection — the internal connections (to children) replace it.
            const toSpec = specMap.get(toId);
            if (toSpec?.type === 'transformer_layer' && expandedLayers?.has(toId))
                continue;
            // 4. If the source is a transformer_layer that IS expanded, remap to its last
            //    child (the residual_add or last in children list) so the line comes from
            //    the bottom of the expanded layer's internals.
            const fromSpec = specMap.get(fromId);
            if (fromSpec?.type === 'transformer_layer' && expandedLayers?.has(fromId)) {
                const kids = fromSpec.children;
                if (kids && kids.length > 0) {
                    // Find the residual_add child, or use the last child
                    const resChild = kids.find(id => specMap.get(id)?.type === 'residual_add');
                    fromId = resChild ?? kids[kids.length - 1];
                }
            }
            const fromNode = nodeMap.get(fromId);
            const toNode = nodeMap.get(toId);
            if (!fromNode || !toNode)
                continue;
            const connGeom = buildConnectionLine(gl, lineProgram, fromNode, toNode, conn, sceneMaxX);
            if (connGeom)
                connections.push(connGeom);
        }
        return { root, nodeMap, connections, totalHeight: cursor };
    }
    function layoutLayerChildren(children, parallel, primitives, expandedAttention) {
        const result = [];
        let localY = 0;
        if (parallel) {
            // Parallel attention layout:
            // Top: layernorms side by side
            // Middle: attention (left) and MLP (right) side by side
            // Bottom: residual add
            const lns = children.filter(c => c.type === 'layernorm');
            const attn = children.find(c => c.type === 'attention');
            const mlp = children.find(c => c.type === 'mlp');
            const resAdd = children.find(c => c.type === 'residual_add');
            const others = children.filter(c => !['layernorm', 'attention', 'mlp', 'residual_add'].includes(c.type));
            // Use wider gap when attention is expanded so heads don't overlap MLP
            const attnExpanded = attn ? expandedAttention?.has(attn.id) === true : false;
            const gap = attnExpanded ? LAYOUT.PARALLEL_GAP_EXPANDED : LAYOUT.PARALLEL_GAP;
            // LayerNorms at top (side by side if two)
            if (lns.length >= 2) {
                const node1 = buildComponentNode(lns[0], primitives);
                result.push({ spec: lns[0], node: node1, x: -gap / 2, y: localY });
                const node2 = buildComponentNode(lns[1], primitives);
                result.push({ spec: lns[1], node: node2, x: gap / 2, y: localY });
                localY -= LAYOUT.LAYERNORM_HEIGHT + LAYOUT.VERTICAL_GAP;
            }
            else if (lns.length === 1) {
                const node = buildComponentNode(lns[0], primitives);
                result.push({ spec: lns[0], node, x: 0, y: localY });
                localY -= LAYOUT.LAYERNORM_HEIGHT + LAYOUT.VERTICAL_GAP;
            }
            // Attention and MLP side by side
            const attnHeight = attnExpanded
                ? getExpandedAttentionHeight(attn?.details.config?.numHeads ?? 4)
                : LAYOUT.BLOCK_HEIGHT;
            const mlpHeight = LAYOUT.BLOCK_HEIGHT;
            const rowHeight = Math.max(attnHeight, mlpHeight);
            localY -= rowHeight / 2;
            if (attn) {
                const node = buildComponentNode(attn, primitives, attnExpanded);
                result.push({ spec: attn, node, x: -gap / 2, y: localY });
            }
            if (mlp) {
                const node = buildComponentNode(mlp, primitives);
                result.push({ spec: mlp, node, x: gap / 2, y: localY });
            }
            localY -= rowHeight / 2 + LAYOUT.VERTICAL_GAP;
            // Residual add at bottom
            if (resAdd) {
                localY -= LAYOUT.DIAMOND_SIZE / 2;
                const node = buildComponentNode(resAdd, primitives);
                result.push({ spec: resAdd, node, x: 0, y: localY });
                localY -= LAYOUT.DIAMOND_SIZE / 2 + LAYOUT.VERTICAL_GAP;
            }
            // Any other components below
            for (const comp of others) {
                const node = buildComponentNode(comp, primitives);
                result.push({ spec: comp, node, x: 0, y: localY });
                localY -= getComponentHeight(comp) + LAYOUT.VERTICAL_GAP;
            }
        }
        else {
            // Sequential layout: stack all children vertically
            for (const comp of children) {
                const attnExpanded = comp.type === 'attention' && expandedAttention?.has(comp.id);
                const node = buildComponentNode(comp, primitives, attnExpanded);
                result.push({ spec: comp, node, x: 0, y: localY });
                const h = attnExpanded
                    ? getExpandedAttentionHeight(comp.details.config?.numHeads ?? 4)
                    : getComponentHeight(comp);
                localY -= h + LAYOUT.VERTICAL_GAP;
            }
        }
        // totalHeight is the positive distance from 0 to the lowest point
        const totalHeight = -localY;
        return { items: result, totalHeight };
    }
    function getComponentHeight(comp) {
        switch (comp.type) {
            case 'embedding': return LAYOUT.EMBEDDING_HEIGHT;
            case 'layernorm': return LAYOUT.LAYERNORM_HEIGHT;
            case 'output_head': return LAYOUT.OUTPUT_HEAD_HEIGHT;
            case 'residual_add': return LAYOUT.DIAMOND_SIZE;
            case 'transformer_layer': return LAYOUT.LAYER_HEIGHT;
            default: return LAYOUT.BLOCK_HEIGHT;
        }
    }
    function getParentLayerId(childId, spec) {
        for (const comp of spec.components) {
            if (comp.type === 'transformer_layer' && comp.children?.includes(childId)) {
                return comp.id;
            }
        }
        return '';
    }
    function buildConnectionLine(gl, lineProgram, fromNode, toNode, conn, sceneMaxX) {
        const fromWorld = getWorldPosition(fromNode);
        const toWorld = getWorldPosition(toNode);
        // Pipeline flows upward: "from" is below (lower Y), "to" is above (higher Y).
        const fromHalfH = fromNode.expanded ? 0 : fromNode.scale[1] * 0.5;
        const fromTopY = fromWorld[1] + fromHalfH + LAYOUT.ARROW_GAP;
        const toHalfH = toNode.expanded ? 0 : toNode.scale[1] * 0.5;
        const toBottomY = toWorld[1] - toHalfH - LAYOUT.ARROW_GAP;
        if (toBottomY < fromTopY - 0.1)
            return null;
        const isResidual = conn.type === 'residual';
        let points;
        if (isResidual) {
            // Route outside ALL scene content — use the scene-wide max X + margin
            const offsetX = sceneMaxX + LAYOUT.RESIDUAL_SPINE_OFFSET;
            points = new Float32Array([
                fromWorld[0], fromTopY, 0,
                offsetX, fromTopY, 0,
                offsetX, toBottomY, 0,
                toWorld[0], toBottomY, 0,
            ]);
        }
        else if (Math.abs(fromWorld[0] - toWorld[0]) < 0.01) {
            points = new Float32Array([
                fromWorld[0], fromTopY, 0,
                toWorld[0], toBottomY, 0,
            ]);
        }
        else {
            const midY = (fromTopY + toBottomY) / 2;
            points = new Float32Array([
                fromWorld[0], fromTopY, 0,
                fromWorld[0], midY, 0,
                toWorld[0], midY, 0,
                toWorld[0], toBottomY, 0,
            ]);
        }
        const geometry = createLineGeometry(gl, lineProgram, points);
        const color = isResidual
            ? new Float32Array([0.55, 0.55, 0.55, 0.4])
            : new Float32Array([0.4, 0.4, 0.5, 0.6]);
        return {
            geometry,
            color,
            fromId: conn.from,
            toId: conn.to,
            type: conn.type ?? 'forward',
        };
    }
    function getWorldPosition(node) {
        let x = node.position[0];
        let y = node.position[1];
        let z = node.position[2];
        let current = node.parent;
        while (current) {
            x += current.position[0];
            y += current.position[1];
            z += current.position[2];
            current = current.parent;
        }
        return new Float32Array([x, y, z]);
    }

    /**
     * Data binding registry: maps activation/gradient hook names to component visual properties.
     * Also stores weight statistics for inspector display and visual coloring.
     */
    /**
     * Create a data binding registry connected to a scene graph node map.
     */
    function createDataBindingRegistry(nodeMap) {
        const bindings = new Map();
        const liveData = new Map();
        const weightDataMap = new Map();
        let runningMaxActivation = 1e-6;
        let runningMaxGradient = 1e-6;
        function bind(hookName, componentId, property) {
            if (!bindings.has(hookName)) {
                bindings.set(hookName, []);
            }
            bindings.get(hookName).push({ componentId, property });
        }
        function unbind(componentId) {
            for (const [hookName, bindingList] of bindings) {
                const filtered = bindingList.filter(b => b.componentId !== componentId);
                if (filtered.length === 0) {
                    bindings.delete(hookName);
                }
                else {
                    bindings.set(hookName, filtered);
                }
            }
            liveData.delete(componentId);
            weightDataMap.delete(componentId);
        }
        function onData(hookName, data, shape) {
            const hookBindings = bindings.get(hookName);
            if (!hookBindings)
                return;
            for (const { componentId, property } of hookBindings) {
                const node = nodeMap.get(componentId);
                if (!node)
                    continue;
                if (!liveData.has(componentId)) {
                    liveData.set(componentId, {});
                }
                const compData = liveData.get(componentId);
                switch (property) {
                    case 'activation': {
                        const norm = l2Norm(data);
                        if (norm > runningMaxActivation)
                            runningMaxActivation = norm;
                        const normalized = norm / runningMaxActivation;
                        node.visualState.activationMagnitude = normalized;
                        compData.activationNorm = norm;
                        break;
                    }
                    case 'gradient': {
                        const norm = l2Norm(data);
                        if (norm > runningMaxGradient)
                            runningMaxGradient = norm;
                        const normalized = norm / runningMaxGradient;
                        node.visualState.gradientMagnitude = normalized;
                        compData.gradientNorm = norm;
                        break;
                    }
                    case 'attention_weights': {
                        let maxVal = 0;
                        for (let i = 0; i < data.length; i++) {
                            if (data[i] > maxVal)
                                maxVal = data[i];
                        }
                        compData.maxAttnWeight = maxVal;
                        node.visualState.activationMagnitude = Math.min(1, maxVal);
                        break;
                    }
                }
            }
        }
        function clear() {
            bindings.clear();
            liveData.clear();
            weightDataMap.clear();
            runningMaxActivation = 1e-6;
            runningMaxGradient = 1e-6;
        }
        function getComponentData(componentId) {
            return liveData.get(componentId);
        }
        // ─── Weight data ─────────────────────────────────────────────────
        function setWeightData(componentId, weightName, data, shape) {
            if (!weightDataMap.has(componentId)) {
                weightDataMap.set(componentId, { weights: [], totalNorm: 0, totalParams: 0 });
            }
            const compWeights = weightDataMap.get(componentId);
            const stats = computeWeightStats(weightName, data, shape);
            compWeights.weights.push(stats);
            // Recompute totals
            let totalNormSq = 0;
            let totalParams = 0;
            for (const w of compWeights.weights) {
                totalNormSq += w.l2Norm * w.l2Norm * w.numParams; // undo RMS, get sum of squares
                totalParams += w.numParams;
            }
            compWeights.totalNorm = Math.sqrt(totalNormSq / Math.max(1, totalParams));
            compWeights.totalParams = totalParams;
        }
        function getWeightData(componentId) {
            return weightDataMap.get(componentId);
        }
        function applyWeightColoring() {
            // Find the max total norm across all components for normalization
            let maxNorm = 1e-8;
            for (const wd of weightDataMap.values()) {
                if (wd.totalNorm > maxNorm)
                    maxNorm = wd.totalNorm;
            }
            for (const [compId, wd] of weightDataMap) {
                const node = nodeMap.get(compId);
                if (!node)
                    continue;
                // Modulate the base color saturation by relative weight norm
                const relNorm = wd.totalNorm / maxNorm; // 0 to 1
                const base = node.visualState.baseColor;
                // Increase saturation: lerp toward a brighter/more saturated version
                // Small weights → desaturated (grayed out), large weights → full color
                const gray = (base[0] + base[1] + base[2]) / 3;
                const satFactor = 0.3 + 0.7 * relNorm; // 0.3 (muted) to 1.0 (full)
                node.visualState.baseColor = [
                    gray + (base[0] - gray) * satFactor,
                    gray + (base[1] - gray) * satFactor,
                    gray + (base[2] - gray) * satFactor,
                    base[3],
                ];
            }
        }
        return {
            bind, unbind, onData, clear, getComponentData,
            setWeightData, getWeightData, applyWeightColoring,
        };
    }
    // ─── Helpers ─────────────────────────────────────────────────────────
    function l2Norm(data) {
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
            sum += data[i] * data[i];
        }
        return Math.sqrt(sum / data.length);
    }
    function computeWeightStats(name, data, shape) {
        const n = data.length;
        let sum = 0;
        let sumSq = 0;
        let min = Infinity;
        let max = -Infinity;
        let sparseCount = 0;
        for (let i = 0; i < n; i++) {
            const v = data[i];
            sum += v;
            sumSq += v * v;
            if (v < min)
                min = v;
            if (v > max)
                max = v;
            if (Math.abs(v) < 1e-6)
                sparseCount++;
        }
        const mean = sum / n;
        const variance = sumSq / n - mean * mean;
        const std = Math.sqrt(Math.max(0, variance));
        const l2 = Math.sqrt(sumSq / n);
        const sparsity = sparseCount / n;
        // 20-bin histogram
        const numBins = 20;
        const bins = [];
        const counts = new Array(numBins).fill(0);
        const range = max - min || 1;
        for (let i = 0; i <= numBins; i++) {
            bins.push(min + (range * i) / numBins);
        }
        for (let i = 0; i < n; i++) {
            let bin = Math.floor(((data[i] - min) / range) * numBins);
            if (bin >= numBins)
                bin = numBins - 1;
            if (bin < 0)
                bin = 0;
            counts[bin]++;
        }
        return {
            name, shape, numParams: n,
            mean, std, min, max, l2Norm: l2, sparsity,
            histogram: { bins, counts, min, max },
        };
    }

    /**
     * Animation controller for data flow visualization.
     * Particles flow along connection paths during forward/backward passes.
     */
    /**
     * Create an animation controller for the given connections.
     */
    function createAnimationController(nodeMap, connections) {
        let _particles = [];
        let _active = false;
        let _direction = 'forward';
        let _totalDuration = 3000;
        let _elapsed = 0;
        // Pre-compute connection order for sequential flow
        // Forward: top to bottom (connections in natural order)
        // Backward: bottom to top (connections in reverse)
        function animateForward(durationMs = 3000) {
            _active = true;
            _direction = 'forward';
            _totalDuration = durationMs;
            _elapsed = 0;
            _particles = [];
            // Create particles, staggered along the pipeline
            const n = connections.length;
            for (let i = 0; i < n; i++) {
                _particles.push({
                    t: 0,
                    x: 0, y: 0, z: 0,
                    color: [0.15, 0.39, 0.92, 0.9], // accent blue
                    connectionIndex: i,
                });
            }
        }
        function animateBackward(durationMs = 3000) {
            _active = true;
            _direction = 'backward';
            _totalDuration = durationMs;
            _elapsed = 0;
            _particles = [];
            const n = connections.length;
            for (let i = 0; i < n; i++) {
                _particles.push({
                    t: 0,
                    x: 0, y: 0, z: 0,
                    color: [0.9, 0.3, 0.1, 0.9], // red-orange for gradients
                    connectionIndex: n - 1 - i, // Reverse order
                });
            }
        }
        function stop() {
            _active = false;
            _particles = [];
            _elapsed = 0;
        }
        function tick(deltaMs) {
            if (!_active || _particles.length === 0)
                return [];
            _elapsed += deltaMs;
            const globalProgress = Math.min(1, _elapsed / _totalDuration);
            const n = _particles.length;
            const stagger = 0.5 / Math.max(1, n); // Stagger particles across time
            for (let i = 0; i < n; i++) {
                const p = _particles[i];
                const startTime = i * stagger;
                const localProgress = Math.max(0, Math.min(1, (globalProgress - startTime) / (1 - n * stagger + stagger)));
                p.t = localProgress;
                // Interpolate position along the connection path
                const conn = connections[p.connectionIndex];
                if (!conn)
                    continue;
                const fromNode = findNodeById(conn.fromId);
                const toNode = findNodeById(conn.toId);
                if (!fromNode || !toNode)
                    continue;
                const fromY = fromNode.worldMatrix[13] - fromNode.scale[1] * 0.5;
                const toY = toNode.worldMatrix[13] + toNode.scale[1] * 0.5;
                if (_direction === 'forward') {
                    p.x = fromNode.worldMatrix[12] + (toNode.worldMatrix[12] - fromNode.worldMatrix[12]) * localProgress;
                    p.y = fromY + (toY - fromY) * localProgress;
                    p.z = 0.5; // Slightly in front
                }
                else {
                    p.x = toNode.worldMatrix[12] + (fromNode.worldMatrix[12] - toNode.worldMatrix[12]) * localProgress;
                    p.y = toY + (fromY - toY) * localProgress;
                    p.z = 0.5;
                }
                // Pulse component activation when particle is near it
                if (localProgress > 0.8 && localProgress < 1.0) {
                    const targetNode = _direction === 'forward' ? toNode : fromNode;
                    if (targetNode) {
                        const pulse = (localProgress - 0.8) / 0.2; // 0 to 1
                        if (_direction === 'forward') {
                            targetNode.visualState.activationMagnitude = Math.max(targetNode.visualState.activationMagnitude, pulse * 0.8);
                        }
                        else {
                            targetNode.visualState.gradientMagnitude = Math.max(targetNode.visualState.gradientMagnitude, pulse * 0.8);
                        }
                    }
                }
            }
            // Auto-stop when animation completes
            if (globalProgress >= 1) {
                // Keep particles visible briefly then fade
                if (_elapsed > _totalDuration + 500) {
                    stop();
                    return [];
                }
            }
            return _particles.filter(p => p.t > 0 && p.t <= 1);
        }
        function findNodeById(id) {
            return nodeMap.get(id);
        }
        return {
            animateForward,
            animateBackward,
            stop,
            tick,
            get isActive() { return _active; },
        };
    }

    /**
     * Main TransformerViz class: orchestrates rendering, interaction, and data binding.
     */
    class TransformerViz {
        dataBindings;
        animation;
        _container;
        _spec;
        _width;
        _height;
        _ctx;
        _camera;
        _primitives;
        _textOverlay;
        _orbitControls;
        _pickTarget;
        _pickingHandler;
        _inspector;
        _layout;
        _expandedLayers;
        _expandedAttention;
        _gap;
        _rafId = 0;
        _lastTime = 0;
        _needsRender = true;
        _selectedNode = null;
        constructor(config) {
            this._container = config.container;
            this._spec = config.spec;
            this._width = config.width ?? 800;
            this._height = config.height ?? 600;
            this._expandedLayers = new Set(config.expandedLayers ?? []);
            this._expandedAttention = new Set();
            this._gap = config.gap ?? LAYOUT.VERTICAL_GAP;
            // Ensure container has relative positioning for overlay elements
            const containerStyle = window.getComputedStyle(this._container);
            if (containerStyle.position === 'static') {
                this._container.style.position = 'relative';
            }
            this._container.style.width = `${this._width}px`;
            this._container.style.height = `${this._height}px`;
            this._container.style.overflow = 'hidden';
            // 1. WebGL context
            this._ctx = createGLContext(this._container, this._width, this._height);
            initAllPrograms(this._ctx);
            // 2. Camera
            this._camera = new OrbitCamera({
                ...(config.cameraPosition && {
                // Apply custom camera position if provided
                }),
            });
            this._camera.setAspect(this._width, this._height);
            if (config.cameraPosition) {
                this._camera.state.azimuth = config.cameraPosition.azimuth;
                this._camera.state.elevation = config.cameraPosition.elevation;
                this._camera.state.distance = config.cameraPosition.distance;
            }
            // 3. Primitives
            const solidProg = this._ctx.programs.get(PROGRAM.SOLID);
            this._primitives = createPrimitives(this._ctx.gl, solidProg);
            // 4. Text overlay
            this._textOverlay = createTextOverlay(this._container, this._width, this._height);
            // 5. Build scene
            this._layout = this._buildScene();
            // 6. Data binding
            this.dataBindings = createDataBindingRegistry(this._layout.nodeMap);
            // 7. Animation
            this.animation = createAnimationController(this._layout.nodeMap, this._layout.connections);
            // 8. Pick target
            const dpr = window.devicePixelRatio || 1;
            this._pickTarget = createPickTarget(this._ctx.gl, Math.round(this._width * dpr), Math.round(this._height * dpr));
            // 9. Inspector
            this._inspector = createInspector(this._container);
            // 10. Orbit controls
            this._orbitControls = createOrbitControls(this._ctx.canvas, this._camera, () => { this._needsRender = true; });
            // 11. Picking
            this._pickingHandler = createPickingHandler(this._ctx, this._camera, this._pickTarget, this._layout.root, {
                onHover: (_node) => { this._needsRender = true; },
                onClick: (node) => { this._onNodeClicked(node); },
            });
            // 12. Start render loop
            this._lastTime = performance.now();
            this._renderLoop();
        }
        /** Expand or collapse a transformer layer. */
        toggleLayer(layerId) {
            if (this._expandedLayers.has(layerId)) {
                this._expandedLayers.delete(layerId);
            }
            else {
                this._expandedLayers.add(layerId);
            }
            this._rebuildScene();
        }
        /** Expand or collapse an attention block to show/hide multi-head internals. */
        toggleAttention(attnId) {
            if (this._expandedAttention.has(attnId)) {
                this._expandedAttention.delete(attnId);
            }
            else {
                this._expandedAttention.add(attnId);
            }
            this._rebuildScene();
        }
        /** Expand all transformer layers. */
        expandAll() {
            for (const comp of this._spec.components) {
                if (comp.type === 'transformer_layer') {
                    this._expandedLayers.add(comp.id);
                }
            }
            this._rebuildScene();
        }
        /** Collapse all transformer layers and attention blocks. */
        collapseAll() {
            this._expandedLayers.clear();
            this._expandedAttention.clear();
            this._rebuildScene();
        }
        /** Set the minimum vertical gap between layers and rebuild. */
        setGap(gap) {
            this._gap = Math.max(0, gap);
            this._rebuildScene();
        }
        /** Get the current gap value. */
        get gap() {
            return this._gap;
        }
        /** Programmatically select a component. */
        select(componentId) {
            const node = this._layout.nodeMap.get(componentId);
            if (node) {
                this._onNodeClicked(node);
            }
        }
        /** Reset camera to default position. */
        resetCamera() {
            this._camera.reset();
            this._needsRender = true;
        }
        /**
         * Load actual model weights into the visualization.
         * Weights are mapped to components by matching weight names declared in the ModelSpec.
         * After loading, components are colored by relative weight norm and the inspector
         * shows per-tensor statistics (mean, std, min, max, L2 norm, sparsity, histogram).
         *
         * @param weights - Map of weight name (e.g. "layers.0.attention.qkv.weight") to
         *                  {data: Float32Array, shape: number[]}.
         * @param nameMapping - Optional map from weight name in `weights` to {componentId, localName}.
         *                      If not provided, the method attempts to auto-match using the
         *                      weight names declared in each ComponentSpec.details.weights.
         */
        loadWeights(weights, nameMapping) {
            if (nameMapping) {
                // Explicit mapping
                for (const [weightName, { componentId, localName }] of nameMapping) {
                    const w = weights.get(weightName);
                    if (w) {
                        this.dataBindings.setWeightData(componentId, localName, w.data, w.shape);
                    }
                }
            }
            else {
                // Auto-match: for each component, check if any of its declared weight names
                // appear as suffixes in the weights map keys
                for (const comp of this._spec.components) {
                    if (!comp.details.weights)
                        continue;
                    for (const localName of Object.keys(comp.details.weights)) {
                        // Try exact match first, then suffix match
                        for (const [weightName, w] of weights) {
                            if (weightName === localName || weightName.endsWith('.' + localName) || weightName.endsWith('/' + localName)) {
                                this.dataBindings.setWeightData(comp.id, localName, w.data, w.shape);
                                break;
                            }
                        }
                    }
                }
            }
            // Apply visual coloring based on weight norms
            this.dataBindings.applyWeightColoring();
            this._needsRender = true;
        }
        /** Update the model spec (e.g., for a different model). */
        setSpec(spec) {
            this._spec = spec;
            this._expandedLayers.clear();
            this._expandedAttention.clear();
            this._rebuildScene();
        }
        /** Clean up all resources. */
        destroy() {
            if (this._rafId) {
                cancelAnimationFrame(this._rafId);
            }
            this._orbitControls.destroy();
            this._pickingHandler.destroy();
            this._inspector.destroy();
            this._textOverlay.destroy();
            destroyPickTarget(this._ctx.gl, this._pickTarget);
            this._ctx.destroy();
        }
        // ─── Internal ──────────────────────────────────────────────────
        _buildScene() {
            resetPickIdCounter();
            const lineProg = this._ctx.programs.get(PROGRAM.LINE);
            return layoutScene(this._spec, this._ctx.gl, lineProg, this._primitives, this._expandedLayers, this._expandedAttention, this._gap);
        }
        _rebuildScene() {
            this._layout = this._buildScene();
            // Reconnect data bindings
            this.dataBindings.clear();
            // Re-bind from spec
            for (const comp of this._spec.components) {
                if (comp.dataBindings) {
                    for (const hookName of comp.dataBindings) {
                        if (hookName.includes('attention_weights')) {
                            this.dataBindings.bind(hookName, comp.id, 'attention_weights');
                        }
                        else {
                            this.dataBindings.bind(hookName, comp.id, 'activation');
                        }
                    }
                }
            }
            // Update picking handler
            this._pickingHandler.updatePickBuffer(this._layout.root);
            // Auto-center camera on the layout
            this._camera.state.focusPoint[1] = this._layout.totalHeight / 2;
            this._needsRender = true;
            this._selectedNode = null;
            this._inspector.hide();
        }
        _onNodeClicked(node) {
            // Clear previous selection
            if (this._selectedNode) {
                this._selectedNode.visualState.selected = false;
            }
            if (!node) {
                this._selectedNode = null;
                this._inspector.hide();
                this._needsRender = true;
                return;
            }
            // Check if it's a transformer layer — toggle expand/collapse
            if (node.componentSpec?.type === 'transformer_layer') {
                this.toggleLayer(node.id);
                return;
            }
            // Check if it's an attention block — toggle multi-head view
            if (node.componentSpec?.type === 'attention') {
                this.toggleAttention(node.id);
                return;
            }
            // Select and show inspector
            node.visualState.selected = true;
            this._selectedNode = node;
            this._inspector.show(node);
            if (node.componentSpec) {
                // Show weight statistics if available
                const weightData = this.dataBindings.getWeightData(node.componentSpec.id);
                if (weightData) {
                    this._inspector.updateWeightData(weightData);
                }
                // Show live activation/gradient data if available
                const liveData = this.dataBindings.getComponentData(node.componentSpec.id);
                if (liveData) {
                    this._inspector.updateLiveData(liveData);
                }
            }
            this._needsRender = true;
        }
        _renderLoop() {
            const now = performance.now();
            const deltaMs = now - this._lastTime;
            this._lastTime = now;
            // Tick animation
            const particles = this.animation.tick(deltaMs);
            if (particles.length > 0 || this.animation.isActive) {
                this._needsRender = true;
            }
            // Decay visual states
            this._decayVisualStates(deltaMs);
            if (this._needsRender) {
                this._needsRender = false;
                // Update transforms
                updateWorldMatrices(this._layout.root);
                // Update pick buffer
                this._pickingHandler.updatePickBuffer(this._layout.root);
                // Render 3D scene
                renderScene(this._ctx, this._camera, this._layout.root, this._layout.connections);
                // Render text overlay
                this._textOverlay.render(this._camera, this._layout.root);
                // Update inspector live data if visible
                if (this._inspector.isVisible && this._selectedNode?.componentSpec) {
                    const liveData = this.dataBindings.getComponentData(this._selectedNode.componentSpec.id);
                    if (liveData) {
                        this._inspector.updateLiveData(liveData);
                    }
                }
            }
            this._rafId = requestAnimationFrame(() => this._renderLoop());
        }
        _decayVisualStates(deltaMs) {
            const decayRate = 1 - Math.min(1, deltaMs * 0.003); // ~300ms half-life
            for (const node of this._layout.nodeMap.values()) {
                if (node.visualState.activationMagnitude > 0.01) {
                    node.visualState.activationMagnitude *= decayRate;
                    this._needsRender = true;
                }
                else {
                    node.visualState.activationMagnitude = 0;
                }
                if (node.visualState.gradientMagnitude > 0.01) {
                    node.visualState.gradientMagnitude *= decayRate;
                    this._needsRender = true;
                }
                else {
                    node.visualState.gradientMagnitude = 0;
                }
            }
        }
    }

    /**
     * Adapter: converts tinyllms PythiaConfig + HookRegistry into the generic ModelSpec format.
     */
    /**
     * Creates a ModelSpec from a PythiaConfig.
     * This is the bridge between tinyllms and transformer-viz.
     */
    function pythiaToModelSpec(config) {
        const components = [];
        const connections = [];
        // Embedding
        components.push({
            id: 'embedding',
            type: 'embedding',
            label: 'Token Embedding',
            details: {
                weights: { 'embed.weight': [config.vocabSize, config.hiddenDim] },
                paramCount: config.vocabSize * config.hiddenDim,
                inputShape: '[seqLen]',
                outputShape: `[seqLen, ${config.hiddenDim}]`,
            },
            dataBindings: ['embedding'],
        });
        let prevId = 'embedding';
        for (let i = 0; i < config.numLayers; i++) {
            const layerId = `layers.${i}`;
            const attnId = `${layerId}.attention`;
            const mlpId = `${layerId}.mlp`;
            const lnInputId = `${layerId}.input_ln`;
            const lnPostId = `${layerId}.post_attn_ln`;
            const residualId = `${layerId}.residual_add`;
            // LayerNorm (input)
            components.push({
                id: lnInputId,
                type: 'layernorm',
                label: 'Input LN',
                details: {
                    paramCount: config.hiddenDim * 2,
                    config: { epsilon: 1e-5 },
                },
            });
            // LayerNorm (post-attn, for parallel MLP)
            components.push({
                id: lnPostId,
                type: 'layernorm',
                label: 'Post-Attn LN',
                details: {
                    paramCount: config.hiddenDim * 2,
                    config: { epsilon: 1e-5 },
                },
            });
            // Attention
            const qkvParams = 3 * config.hiddenDim * config.hiddenDim + 3 * config.hiddenDim;
            const outParams = config.hiddenDim * config.hiddenDim + config.hiddenDim;
            components.push({
                id: attnId,
                type: 'attention',
                label: 'Self-Attention',
                details: {
                    weights: {
                        'qkv.weight': [3 * config.hiddenDim, config.hiddenDim],
                        'qkv.bias': [3 * config.hiddenDim],
                        'out.weight': [config.hiddenDim, config.hiddenDim],
                        'out.bias': [config.hiddenDim],
                    },
                    paramCount: qkvParams + outParams,
                    inputShape: `[seqLen, ${config.hiddenDim}]`,
                    outputShape: `[seqLen, ${config.hiddenDim}]`,
                    config: {
                        numHeads: config.numHeads,
                        headDim: config.headDim,
                        usesRoPE: true,
                        causal: true,
                    },
                },
                dataBindings: [`layers.${i}.attention_weights`],
            });
            // MLP
            const mlpUpParams = config.hiddenDim * config.intermediateSize + config.intermediateSize;
            const mlpDownParams = config.intermediateSize * config.hiddenDim + config.hiddenDim;
            components.push({
                id: mlpId,
                type: 'mlp',
                label: 'MLP',
                details: {
                    weights: {
                        'dense_h_to_4h.weight': [config.intermediateSize, config.hiddenDim],
                        'dense_h_to_4h.bias': [config.intermediateSize],
                        'dense_4h_to_h.weight': [config.hiddenDim, config.intermediateSize],
                        'dense_4h_to_h.bias': [config.hiddenDim],
                    },
                    paramCount: mlpUpParams + mlpDownParams,
                    config: { intermediateSize: config.intermediateSize, activation: 'gelu' },
                },
            });
            // Residual add
            components.push({
                id: residualId,
                type: 'residual_add',
                label: '+',
                details: {},
            });
            // Transformer layer (container)
            components.push({
                id: layerId,
                type: 'transformer_layer',
                label: `Layer ${i}`,
                children: [lnInputId, lnPostId, attnId, mlpId, residualId],
                details: {
                    config: { parallel: true }, // Pythia uses parallel attention
                },
                dataBindings: [`${layerId}.output`],
            });
            // Connections within the layer
            connections.push({ from: prevId, to: layerId, type: 'forward' });
            // Internal connections (visible when expanded)
            connections.push({ from: prevId, to: lnInputId, type: 'forward' });
            connections.push({ from: prevId, to: lnPostId, type: 'forward' });
            connections.push({ from: lnInputId, to: attnId, type: 'forward' });
            connections.push({ from: lnPostId, to: mlpId, type: 'forward' });
            connections.push({ from: attnId, to: residualId, type: 'forward' });
            connections.push({ from: mlpId, to: residualId, type: 'forward' });
            connections.push({ from: prevId, to: residualId, type: 'residual' });
            prevId = layerId;
        }
        // Final LayerNorm
        components.push({
            id: 'final_ln',
            type: 'layernorm',
            label: 'Final LN',
            details: { paramCount: config.hiddenDim * 2 },
        });
        connections.push({ from: prevId, to: 'final_ln', type: 'forward' });
        // LM Head
        components.push({
            id: 'lm_head',
            type: 'output_head',
            label: 'LM Head',
            details: {
                weights: { 'lm_head.weight': [config.vocabSize, config.hiddenDim] },
                paramCount: config.vocabSize * config.hiddenDim,
                outputShape: `[seqLen, ${config.vocabSize}]`,
            },
            dataBindings: ['logits'],
        });
        connections.push({ from: 'final_ln', to: 'lm_head', type: 'forward' });
        // Metadata
        let totalParams = 0;
        for (const comp of components) {
            if (comp.details.paramCount)
                totalParams += comp.details.paramCount;
        }
        return {
            name: `Pythia (${config.hiddenDim}d, ${config.numLayers}L)`,
            components,
            connections,
            metadata: { totalParams },
        };
    }
    /**
     * Connects a tinyllms HookRegistry to the transformer-viz DataBindingRegistry.
     * Returns a cleanup function that removes all hooks.
     */
    function connectHooks(hooks, bindings) {
        const remover = hooks.add('*', (name, tensor) => {
            bindings.onData(name, tensor.contiguousData(), [...tensor.shape]);
        });
        return remover;
    }
    /**
     * Build a weight name mapping from tinyllms weight names to component IDs.
     * tinyllms weights use names like "layers.0.attention.qkv.weight",
     * which need to be mapped to component IDs like "layers.0.attention"
     * with local names like "qkv.weight".
     */
    function buildPythiaWeightMapping(config) {
        const mapping = new Map();
        // Embedding
        mapping.set('embed.weight', { componentId: 'embedding', localName: 'embed.weight' });
        for (let i = 0; i < config.numLayers; i++) {
            const prefix = `layers.${i}`;
            // Attention weights
            mapping.set(`${prefix}.attention.qkv.weight`, { componentId: `${prefix}.attention`, localName: 'qkv.weight' });
            mapping.set(`${prefix}.attention.qkv.bias`, { componentId: `${prefix}.attention`, localName: 'qkv.bias' });
            mapping.set(`${prefix}.attention.out.weight`, { componentId: `${prefix}.attention`, localName: 'out.weight' });
            mapping.set(`${prefix}.attention.out.bias`, { componentId: `${prefix}.attention`, localName: 'out.bias' });
            // MLP weights
            mapping.set(`${prefix}.mlp.dense_h_to_4h.weight`, { componentId: `${prefix}.mlp`, localName: 'dense_h_to_4h.weight' });
            mapping.set(`${prefix}.mlp.dense_h_to_4h.bias`, { componentId: `${prefix}.mlp`, localName: 'dense_h_to_4h.bias' });
            mapping.set(`${prefix}.mlp.dense_4h_to_h.weight`, { componentId: `${prefix}.mlp`, localName: 'dense_4h_to_h.weight' });
            mapping.set(`${prefix}.mlp.dense_4h_to_h.bias`, { componentId: `${prefix}.mlp`, localName: 'dense_4h_to_h.bias' });
            // LayerNorm weights
            mapping.set(`${prefix}.input_ln.weight`, { componentId: `${prefix}.input_ln`, localName: 'weight' });
            mapping.set(`${prefix}.input_ln.bias`, { componentId: `${prefix}.input_ln`, localName: 'bias' });
            mapping.set(`${prefix}.post_attn_ln.weight`, { componentId: `${prefix}.post_attn_ln`, localName: 'weight' });
            mapping.set(`${prefix}.post_attn_ln.bias`, { componentId: `${prefix}.post_attn_ln`, localName: 'bias' });
        }
        // Final LayerNorm
        mapping.set('final_ln.weight', { componentId: 'final_ln', localName: 'weight' });
        mapping.set('final_ln.bias', { componentId: 'final_ln', localName: 'bias' });
        // LM Head
        mapping.set('lm_head.weight', { componentId: 'lm_head', localName: 'lm_head.weight' });
        return mapping;
    }
    /**
     * Auto-bind all data bindings declared in the model spec.
     */
    function autoBindSpec(spec, bindings) {
        for (const comp of spec.components) {
            if (comp.dataBindings) {
                for (const hookName of comp.dataBindings) {
                    // Infer property from hook name
                    if (hookName.includes('attention_weights')) {
                        bindings.bind(hookName, comp.id, 'attention_weights');
                    }
                    else if (hookName.includes('gradient') || hookName.includes('grad')) {
                        bindings.bind(hookName, comp.id, 'gradient');
                    }
                    else {
                        bindings.bind(hookName, comp.id, 'activation');
                    }
                }
            }
        }
    }

    exports.COMPONENT_COLORS = COMPONENT_COLORS;
    exports.LAYOUT = LAYOUT;
    exports.TransformerViz = TransformerViz;
    exports.autoBindSpec = autoBindSpec;
    exports.buildPythiaWeightMapping = buildPythiaWeightMapping;
    exports.connectHooks = connectHooks;
    exports.pythiaToModelSpec = pythiaToModelSpec;

}));
//# sourceMappingURL=transformer-viz.umd.js.map
