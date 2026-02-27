(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.tinyllms = {}));
})(this, (function (exports) { 'use strict';

    function assert(condition, msg) {
        if (!condition) {
            throw new Error(`[tinyllms] ${msg}`);
        }
    }
    function shapeSize(shape) {
        let size = 1;
        for (const d of shape) {
            size *= d;
        }
        return size;
    }
    function shapeEqual(a, b) {
        if (a.length !== b.length)
            return false;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i])
                return false;
        }
        return true;
    }
    function computeStrides(shape) {
        const strides = new Array(shape.length);
        let stride = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
    function offsetFor(indices, strides) {
        let offset = 0;
        for (let i = 0; i < indices.length; i++) {
            offset += indices[i] * strides[i];
        }
        return offset;
    }
    function inferShape(data) {
        if (!Array.isArray(data))
            return [];
        const shape = [data.length];
        let current = data[0];
        while (Array.isArray(current)) {
            shape.push(current.length);
            current = current[0];
        }
        return shape;
    }
    function isContiguous(shape, strides) {
        let expected = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            if (strides[i] !== expected)
                return false;
            expected *= shape[i];
        }
        return true;
    }
    function contiguousData(data, shape, strides) {
        if (isContiguous(shape, strides))
            return data;
        const size = shapeSize(shape);
        const result = new Float32Array(size);
        const ndim = shape.length;
        const indices = new Array(ndim).fill(0);
        computeStrides(shape);
        for (let i = 0; i < size; i++) {
            let srcOffset = 0;
            for (let d = 0; d < ndim; d++) {
                srcOffset += indices[d] * strides[d];
            }
            result[i] = data[srcOffset];
            // Increment indices (last dimension first)
            for (let d = ndim - 1; d >= 0; d--) {
                indices[d]++;
                if (indices[d] < shape[d])
                    break;
                indices[d] = 0;
            }
        }
        return result;
    }
    /** Flatten nested number arrays into a Float32Array */
    function flattenToFloat32(data) {
        const flat = [];
        const recurse = (d) => {
            if (Array.isArray(d)) {
                for (const item of d)
                    recurse(item);
            }
            else if (typeof d === 'number') {
                flat.push(d);
            }
        };
        recurse(data);
        return new Float32Array(flat);
    }

    class Tensor {
        data;
        shape;
        strides;
        size;
        requiresGrad;
        grad;
        _backward;
        _children;
        constructor(data, shape, strides, requiresGrad = false) {
            this.data = data;
            this.shape = shape;
            this.strides = strides ?? computeStrides(shape);
            this.size = shapeSize(shape);
            this.requiresGrad = requiresGrad;
            this.grad = null;
            this._backward = null;
            this._children = [];
        }
        // --- Static factories ---
        static zeros(shape, requiresGrad = false) {
            return new Tensor(new Float32Array(shapeSize(shape)), shape, undefined, requiresGrad);
        }
        static ones(shape, requiresGrad = false) {
            const data = new Float32Array(shapeSize(shape));
            data.fill(1);
            return new Tensor(data, shape, undefined, requiresGrad);
        }
        static full(shape, value, requiresGrad = false) {
            const data = new Float32Array(shapeSize(shape));
            data.fill(value);
            return new Tensor(data, shape, undefined, requiresGrad);
        }
        static rand(shape, requiresGrad = false) {
            const size = shapeSize(shape);
            const data = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = Math.random();
            }
            return new Tensor(data, shape, undefined, requiresGrad);
        }
        static randn(shape, requiresGrad = false) {
            const size = shapeSize(shape);
            const data = new Float32Array(size);
            // Box-Muller transform
            for (let i = 0; i < size; i += 2) {
                const u1 = Math.random() || 1e-10;
                const u2 = Math.random();
                const mag = Math.sqrt(-2 * Math.log(u1));
                data[i] = mag * Math.cos(2 * Math.PI * u2);
                if (i + 1 < size) {
                    data[i + 1] = mag * Math.sin(2 * Math.PI * u2);
                }
            }
            return new Tensor(data, shape, undefined, requiresGrad);
        }
        static scalar(value, requiresGrad = false) {
            return new Tensor(new Float32Array([value]), [], undefined, requiresGrad);
        }
        static from(data, requiresGrad = false) {
            if (typeof data === 'number') {
                return Tensor.scalar(data, requiresGrad);
            }
            if (data instanceof Float32Array) {
                return new Tensor(new Float32Array(data), [data.length], undefined, requiresGrad);
            }
            const shape = inferShape(data);
            const flat = flattenToFloat32(data);
            return new Tensor(flat, shape, undefined, requiresGrad);
        }
        // --- Data access ---
        /** Get the scalar value. Only valid for scalar tensors (size 1). */
        item() {
            assert(this.size === 1, `item() requires size 1, got size ${this.size}`);
            return this.data[offsetFor(new Array(this.shape.length).fill(0), this.strides)];
        }
        /** Get value at given indices. */
        get(...indices) {
            assert(indices.length === this.shape.length, `Expected ${this.shape.length} indices, got ${indices.length}`);
            return this.data[offsetFor(indices, this.strides)];
        }
        /** Set value at given indices. */
        set(indices, value) {
            assert(indices.length === this.shape.length, `Expected ${this.shape.length} indices, got ${indices.length}`);
            this.data[offsetFor(indices, this.strides)] = value;
        }
        /** Return flat contiguous data as a regular Array. */
        toArray() {
            const d = this.contiguousData();
            return Array.from(d);
        }
        /** Return nested array matching the tensor's shape. */
        toNestedArray() {
            if (this.shape.length === 0)
                return this.item();
            const d = this.contiguousData();
            const buildNested = (offset, dim) => {
                if (dim === this.shape.length - 1) {
                    const size = this.shape[dim];
                    return Array.from(d.subarray(offset, offset + size));
                }
                const result = [];
                const innerSize = shapeSize(this.shape.slice(dim + 1));
                for (let i = 0; i < this.shape[dim]; i++) {
                    result.push(buildNested(offset + i * innerSize, dim + 1));
                }
                return result;
            };
            return buildNested(0, 0);
        }
        /** Get contiguous Float32Array data (copies if non-contiguous). */
        contiguousData() {
            return contiguousData(this.data, this.shape, this.strides);
        }
        /** Return true if this tensor is contiguous in memory. */
        isContiguous() {
            return isContiguous(this.shape, this.strides);
        }
        /** Detach from autograd graph. Returns a new tensor sharing the same data. */
        detach() {
            return new Tensor(this.data, this.shape, this.strides, false);
        }
        /** Number of dimensions. */
        get ndim() {
            return this.shape.length;
        }
        // --- Convenience methods (delegate to ops) ---
        // These are defined as stubs here. The actual ops module will provide
        // the implementations, and we wire them up after import.
        add(other) {
            // Will be wired by ops.ts
            throw new Error('[tinyllms] ops not initialized');
        }
        mul(other) {
            throw new Error('[tinyllms] ops not initialized');
        }
        matmul(other) {
            throw new Error('[tinyllms] ops not initialized');
        }
        reshape(shape) {
            throw new Error('[tinyllms] ops not initialized');
        }
        transpose(dim0, dim1) {
            throw new Error('[tinyllms] ops not initialized');
        }
        sum(dim, keepdim) {
            throw new Error('[tinyllms] ops not initialized');
        }
        mean(dim, keepdim) {
            throw new Error('[tinyllms] ops not initialized');
        }
        neg() {
            throw new Error('[tinyllms] ops not initialized');
        }
        /** Backward pass. Only valid on scalar tensors. */
        backward() {
            // Will be wired by autograd.ts
            throw new Error('[tinyllms] autograd not initialized');
        }
    }

    /**
     * Reverse-mode automatic differentiation.
     * Performs topological sort via DFS post-order, then calls _backward
     * in reverse topological order. Cleans up graph after completion.
     */
    function backward(tensor) {
        assert(tensor.size === 1, `backward() can only be called on scalar tensors, got shape [${tensor.shape}]`);
        // Seed gradient
        tensor.grad = new Tensor(new Float32Array([1]), tensor.shape);
        // Topological sort (DFS post-order)
        const sorted = [];
        const visited = new Set();
        function dfs(t) {
            if (visited.has(t))
                return;
            visited.add(t);
            for (const child of t._children) {
                dfs(child);
            }
            sorted.push(t);
        }
        dfs(tensor);
        // Walk in reverse topological order (from output to inputs)
        for (let i = sorted.length - 1; i >= 0; i--) {
            const t = sorted[i];
            if (t._backward) {
                t._backward();
            }
        }
        // Cleanup: null out _backward and clear _children
        for (const t of sorted) {
            t._backward = null;
            t._children = [];
        }
    }
    // Wire backward onto Tensor prototype
    Tensor.prototype.backward = function () {
        backward(this);
    };

    /**
     * Compute the broadcast shape of two shapes following NumPy rules.
     * Shapes are right-aligned; dimensions must be equal, or one must be 1.
     */
    function broadcastShape(a, b) {
        const ndim = Math.max(a.length, b.length);
        const result = new Array(ndim);
        for (let i = 0; i < ndim; i++) {
            const da = i < a.length ? a[a.length - 1 - i] : 1;
            const db = i < b.length ? b[b.length - 1 - i] : 1;
            assert(da === db || da === 1 || db === 1, `Shapes [${a}] and [${b}] are not broadcastable`);
            result[ndim - 1 - i] = Math.max(da, db);
        }
        return result;
    }
    /**
     * Given an original shape and the broadcast shape it was expanded to,
     * return the axes along which the original shape had size 1 (or was absent).
     * These are the axes to sum over in the backward pass.
     */
    function reductionAxes(original, broadcast) {
        const axes = [];
        const offset = broadcast.length - original.length;
        for (let i = 0; i < broadcast.length; i++) {
            const origDim = i < offset ? 1 : original[i - offset];
            if (origDim === 1 && broadcast[i] > 1) {
                axes.push(i);
            }
        }
        // Also include leading dimensions that original didn't have
        for (let i = 0; i < offset; i++) {
            if (!axes.includes(i))
                axes.push(i);
        }
        return axes.sort((a, b) => a - b);
    }
    /**
     * Compute the flat index into a tensor's data given a multi-dimensional
     * index into a broadcast shape. For dimensions where the original shape
     * is 1, the index wraps to 0 (broadcast behavior).
     */
    function broadcastIndex(broadcastIdx, originalShape, originalStrides) {
        const offset = broadcastIdx.length - originalShape.length;
        let idx = 0;
        for (let i = 0; i < originalShape.length; i++) {
            const dim = originalShape[i];
            const bIdx = broadcastIdx[i + offset];
            idx += (dim === 1 ? 0 : bIdx) * originalStrides[i];
        }
        return idx;
    }

    // ─── Helpers ──────────────────────────────────────────────────────────
    function needsGrad(...tensors) {
        return tensors.some((t) => t.requiresGrad);
    }
    function accumGrad(tensor, gradData) {
        if (!tensor.grad) {
            tensor.grad = new Tensor(gradData, [...tensor.shape]);
        }
        else {
            const existing = tensor.grad.contiguousData();
            for (let i = 0; i < existing.length; i++) {
                existing[i] += gradData[i];
            }
            tensor.grad = new Tensor(existing, [...tensor.shape]);
        }
    }
    /**
     * Sum a Float32Array along specified axes of a given shape.
     * Returns { data, shape } of the reduced tensor.
     */
    function sumAlongAxes(data, shape, axes) {
        if (axes.length === 0)
            return { data: new Float32Array(data), shape: [...shape] };
        const axisSet = new Set(axes);
        const outShape = shape.map((d, i) => (axisSet.has(i) ? 1 : d));
        const outSize = shapeSize(outShape);
        const result = new Float32Array(outSize);
        const outStrides = computeStrides(outShape);
        const ndim = shape.length;
        const totalSize = shapeSize(shape);
        const indices = new Array(ndim).fill(0);
        for (let i = 0; i < totalSize; i++) {
            let outIdx = 0;
            for (let d = 0; d < ndim; d++) {
                outIdx += (axisSet.has(d) ? 0 : indices[d]) * outStrides[d];
            }
            result[outIdx] += data[i];
            for (let d = ndim - 1; d >= 0; d--) {
                indices[d]++;
                if (indices[d] < shape[d])
                    break;
                indices[d] = 0;
            }
        }
        const squeezedShape = outShape.filter((_, i) => !axisSet.has(i));
        return { data: result, shape: squeezedShape.length > 0 ? squeezedShape : [] };
    }
    /** Iterate N-d indices in row-major order, incrementing in place. */
    function incrementIndices(indices, shape) {
        for (let d = shape.length - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < shape[d])
                return;
            indices[d] = 0;
        }
    }
    // ─── Arithmetic Ops ───────────────────────────────────────────────────
    function add(a, b) {
        const outShape = broadcastShape(a.shape, b.shape);
        const outSize = shapeSize(outShape);
        const data = new Float32Array(outSize);
        const ndim = outShape.length;
        const indices = new Array(ndim).fill(0);
        const aContStrides = computeStrides([...a.shape]);
        const bContStrides = computeStrides([...b.shape]);
        const aData = a.contiguousData();
        const bData = b.contiguousData();
        for (let i = 0; i < outSize; i++) {
            const ai = broadcastIndex(indices, [...a.shape], aContStrides);
            const bi = broadcastIndex(indices, [...b.shape], bContStrides);
            data[i] = aData[ai] + bData[bi];
            incrementIndices(indices, outShape);
        }
        const g = needsGrad(a, b);
        const result = new Tensor(data, outShape, undefined, g);
        if (g) {
            result._children = [a, b].filter((t) => t.requiresGrad);
            result._backward = () => {
                const gData = result.grad.contiguousData();
                if (a.requiresGrad) {
                    const axes = reductionAxes(a.shape, outShape);
                    const reduced = axes.length > 0 ? sumAlongAxes(gData, outShape, axes).data : new Float32Array(gData);
                    accumGrad(a, reduced);
                }
                if (b.requiresGrad) {
                    const axes = reductionAxes(b.shape, outShape);
                    const reduced = axes.length > 0 ? sumAlongAxes(gData, outShape, axes).data : new Float32Array(gData);
                    accumGrad(b, reduced);
                }
            };
        }
        return result;
    }
    function mul(a, b) {
        const outShape = broadcastShape(a.shape, b.shape);
        const outSize = shapeSize(outShape);
        const data = new Float32Array(outSize);
        const ndim = outShape.length;
        const indices = new Array(ndim).fill(0);
        const aData = a.contiguousData();
        const bData = b.contiguousData();
        const aContStrides = computeStrides([...a.shape]);
        const bContStrides = computeStrides([...b.shape]);
        for (let i = 0; i < outSize; i++) {
            const ai = broadcastIndex(indices, [...a.shape], aContStrides);
            const bi = broadcastIndex(indices, [...b.shape], bContStrides);
            data[i] = aData[ai] * bData[bi];
            incrementIndices(indices, outShape);
        }
        const g = needsGrad(a, b);
        const result = new Tensor(data, outShape, undefined, g);
        if (g) {
            result._children = [a, b].filter((t) => t.requiresGrad);
            result._backward = () => {
                const gData = result.grad.contiguousData();
                if (a.requiresGrad) {
                    const gradA = new Float32Array(outSize);
                    const idx = new Array(ndim).fill(0);
                    for (let i = 0; i < outSize; i++) {
                        const bi = broadcastIndex(idx, [...b.shape], bContStrides);
                        gradA[i] = gData[i] * bData[bi];
                        incrementIndices(idx, outShape);
                    }
                    const axes = reductionAxes(a.shape, outShape);
                    const reduced = axes.length > 0 ? sumAlongAxes(gradA, outShape, axes).data : gradA;
                    accumGrad(a, reduced);
                }
                if (b.requiresGrad) {
                    const gradB = new Float32Array(outSize);
                    const idx = new Array(ndim).fill(0);
                    for (let i = 0; i < outSize; i++) {
                        const ai = broadcastIndex(idx, [...a.shape], aContStrides);
                        gradB[i] = gData[i] * aData[ai];
                        incrementIndices(idx, outShape);
                    }
                    const axes = reductionAxes(b.shape, outShape);
                    const reduced = axes.length > 0 ? sumAlongAxes(gradB, outShape, axes).data : gradB;
                    accumGrad(b, reduced);
                }
            };
        }
        return result;
    }
    function neg(a) {
        const aData = a.contiguousData();
        const data = new Float32Array(a.size);
        for (let i = 0; i < a.size; i++)
            data[i] = -aData[i];
        const result = new Tensor(data, [...a.shape], undefined, a.requiresGrad);
        if (a.requiresGrad) {
            result._children = [a];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradA = new Float32Array(a.size);
                for (let i = 0; i < a.size; i++)
                    gradA[i] = -gData[i];
                accumGrad(a, gradA);
            };
        }
        return result;
    }
    function sub(a, b) {
        return add(a, neg(b));
    }
    function div(a, b) {
        const outShape = broadcastShape(a.shape, b.shape);
        const outSize = shapeSize(outShape);
        const data = new Float32Array(outSize);
        const ndim = outShape.length;
        const indices = new Array(ndim).fill(0);
        const aData = a.contiguousData();
        const bData = b.contiguousData();
        const aContStrides = computeStrides([...a.shape]);
        const bContStrides = computeStrides([...b.shape]);
        for (let i = 0; i < outSize; i++) {
            const ai = broadcastIndex(indices, [...a.shape], aContStrides);
            const bi = broadcastIndex(indices, [...b.shape], bContStrides);
            data[i] = aData[ai] / bData[bi];
            incrementIndices(indices, outShape);
        }
        const g = needsGrad(a, b);
        const result = new Tensor(data, outShape, undefined, g);
        if (g) {
            result._children = [a, b].filter((t) => t.requiresGrad);
            result._backward = () => {
                const gData = result.grad.contiguousData();
                if (a.requiresGrad) {
                    const gradA = new Float32Array(outSize);
                    const idx = new Array(ndim).fill(0);
                    for (let i = 0; i < outSize; i++) {
                        const bi = broadcastIndex(idx, [...b.shape], bContStrides);
                        gradA[i] = gData[i] / bData[bi];
                        incrementIndices(idx, outShape);
                    }
                    const axes = reductionAxes(a.shape, outShape);
                    const reduced = axes.length > 0 ? sumAlongAxes(gradA, outShape, axes).data : gradA;
                    accumGrad(a, reduced);
                }
                if (b.requiresGrad) {
                    const gradB = new Float32Array(outSize);
                    const idx = new Array(ndim).fill(0);
                    for (let i = 0; i < outSize; i++) {
                        const ai = broadcastIndex(idx, [...a.shape], aContStrides);
                        const bi = broadcastIndex(idx, [...b.shape], bContStrides);
                        gradB[i] = -gData[i] * aData[ai] / (bData[bi] * bData[bi]);
                        incrementIndices(idx, outShape);
                    }
                    const axes = reductionAxes(b.shape, outShape);
                    const reduced = axes.length > 0 ? sumAlongAxes(gradB, outShape, axes).data : gradB;
                    accumGrad(b, reduced);
                }
            };
        }
        return result;
    }
    // ─── Reduction Ops ────────────────────────────────────────────────────
    function sum(a, dim, keepdim = false) {
        const aData = a.contiguousData();
        if (dim === undefined) {
            let s = 0;
            for (let i = 0; i < aData.length; i++)
                s += aData[i];
            const result = new Tensor(new Float32Array([s]), [], undefined, a.requiresGrad);
            if (a.requiresGrad) {
                result._children = [a];
                result._backward = () => {
                    const g = result.grad.data[0];
                    const gradA = new Float32Array(a.size);
                    gradA.fill(g);
                    accumGrad(a, gradA);
                };
            }
            return result;
        }
        const ndim = a.shape.length;
        const axis = dim < 0 ? ndim + dim : dim;
        assert(axis >= 0 && axis < ndim, `dim ${dim} out of range for ${ndim}D tensor`);
        const outShape = [];
        for (let i = 0; i < ndim; i++) {
            if (i === axis) {
                if (keepdim)
                    outShape.push(1);
            }
            else {
                outShape.push(a.shape[i]);
            }
        }
        const outSize = shapeSize(outShape.length > 0 ? outShape : [1]);
        const outData = new Float32Array(outSize);
        const outStrides = computeStrides(outShape.length > 0 ? outShape : [1]);
        const indices = new Array(ndim).fill(0);
        for (let i = 0; i < a.size; i++) {
            const outIdx = [];
            for (let d = 0; d < ndim; d++) {
                if (d === axis) {
                    if (keepdim)
                        outIdx.push(0);
                }
                else {
                    outIdx.push(indices[d]);
                }
            }
            let oi = 0;
            for (let d = 0; d < outIdx.length; d++) {
                oi += outIdx[d] * outStrides[d];
            }
            outData[oi] += aData[i];
            incrementIndices(indices, [...a.shape]);
        }
        const finalShape = outShape.length > 0 ? outShape : [];
        const result = new Tensor(outData, finalShape, undefined, a.requiresGrad);
        if (a.requiresGrad) {
            result._children = [a];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradA = new Float32Array(a.size);
                const idx = new Array(ndim).fill(0);
                for (let i = 0; i < a.size; i++) {
                    const outIdx = [];
                    for (let d = 0; d < ndim; d++) {
                        if (d === axis) {
                            if (keepdim)
                                outIdx.push(0);
                        }
                        else {
                            outIdx.push(idx[d]);
                        }
                    }
                    let oi = 0;
                    for (let d = 0; d < outIdx.length; d++) {
                        oi += outIdx[d] * outStrides[d];
                    }
                    gradA[i] = gData[oi];
                    incrementIndices(idx, [...a.shape]);
                }
                accumGrad(a, gradA);
            };
        }
        return result;
    }
    function mean(a, dim, keepdim = false) {
        const s = sum(a, dim, keepdim);
        const count = dim === undefined ? a.size : a.shape[dim < 0 ? a.shape.length + dim : dim];
        const countTensor = Tensor.scalar(count);
        return div(s, countTensor);
    }
    function max(a, dim) {
        const aData = a.contiguousData();
        const ndim = a.shape.length;
        const axis = dim < 0 ? ndim + dim : dim;
        assert(axis >= 0 && axis < ndim, `dim ${dim} out of range for ${ndim}D tensor`);
        const outShape = [];
        for (let i = 0; i < ndim; i++) {
            if (i !== axis)
                outShape.push(a.shape[i]);
        }
        const outSize = shapeSize(outShape.length > 0 ? outShape : [1]);
        const outData = new Float32Array(outSize);
        outData.fill(-Infinity);
        const maxIndices = new Array(outSize).fill(0);
        const outStrides = computeStrides(outShape.length > 0 ? outShape : [1]);
        const indices = new Array(ndim).fill(0);
        for (let i = 0; i < a.size; i++) {
            const outIdx = [];
            for (let d = 0; d < ndim; d++) {
                if (d !== axis)
                    outIdx.push(indices[d]);
            }
            let oi = 0;
            for (let d = 0; d < outIdx.length; d++) {
                oi += outIdx[d] * outStrides[d];
            }
            if (aData[i] > outData[oi]) {
                outData[oi] = aData[i];
                maxIndices[oi] = indices[axis];
            }
            incrementIndices(indices, [...a.shape]);
        }
        const result = new Tensor(outData, outShape, undefined, false);
        return { values: result, indices: maxIndices };
    }
    // ─── Shape Ops ────────────────────────────────────────────────────────
    function reshape(a, shape) {
        const newSize = shapeSize(shape);
        assert(newSize === a.size, `Cannot reshape tensor of size ${a.size} to shape [${shape}]`);
        const aData = a.contiguousData();
        const result = new Tensor(aData, shape, undefined, a.requiresGrad);
        if (a.requiresGrad) {
            result._children = [a];
            result._backward = () => {
                accumGrad(a, result.grad.contiguousData());
            };
        }
        return result;
    }
    function transpose(a, dim0, dim1) {
        const ndim = a.shape.length;
        assert(dim0 >= 0 && dim0 < ndim, `dim0 ${dim0} out of range`);
        assert(dim1 >= 0 && dim1 < ndim, `dim1 ${dim1} out of range`);
        const newShape = [...a.shape];
        const newStrides = [...a.strides];
        [newShape[dim0], newShape[dim1]] = [newShape[dim1], newShape[dim0]];
        [newStrides[dim0], newStrides[dim1]] = [newStrides[dim1], newStrides[dim0]];
        const result = new Tensor(a.data, newShape, newStrides, a.requiresGrad);
        if (a.requiresGrad) {
            result._children = [a];
            result._backward = () => {
                // Transpose grad back: swap the same dims
                const gContiguous = result.grad.contiguousData();
                // gContiguous is in the transposed layout. We need to transpose it back.
                const gShape = [...result.grad.shape];
                const srcSize = shapeSize(gShape);
                const dstData = new Float32Array(srcSize);
                computeStrides(gShape);
                const aShapeArr = [...a.shape];
                const dstStrides = computeStrides(aShapeArr);
                const srcIdx = new Array(ndim).fill(0);
                for (let i = 0; i < srcSize; i++) {
                    // Map from grad (transposed) indices to original indices
                    const dstIdx = [...srcIdx];
                    [dstIdx[dim0], dstIdx[dim1]] = [dstIdx[dim1], dstIdx[dim0]];
                    let dstOffset = 0;
                    for (let d = 0; d < ndim; d++) {
                        dstOffset += dstIdx[d] * dstStrides[d];
                    }
                    dstData[dstOffset] = gContiguous[i];
                    incrementIndices(srcIdx, gShape);
                }
                accumGrad(a, dstData);
            };
        }
        return result;
    }
    function slice(a, starts, ends) {
        const ndim = a.shape.length;
        assert(starts.length === ndim, `starts must have ${ndim} elements`);
        assert(ends.length === ndim, `ends must have ${ndim} elements`);
        const outShape = [];
        for (let i = 0; i < ndim; i++) {
            outShape.push(ends[i] - starts[i]);
            assert(outShape[i] > 0, `Slice dimension ${i} is empty`);
            assert(starts[i] >= 0 && ends[i] <= a.shape[i], `Slice out of bounds on dim ${i}`);
        }
        const outSize = shapeSize(outShape);
        const data = new Float32Array(outSize);
        const aStrides = a.isContiguous() ? computeStrides([...a.shape]) : [...a.strides];
        const outIndices = new Array(ndim).fill(0);
        for (let i = 0; i < outSize; i++) {
            let srcOffset = 0;
            for (let d = 0; d < ndim; d++) {
                srcOffset += (starts[d] + outIndices[d]) * aStrides[d];
            }
            data[i] = a.data[srcOffset];
            incrementIndices(outIndices, outShape);
        }
        const result = new Tensor(data, outShape, undefined, a.requiresGrad);
        if (a.requiresGrad) {
            result._children = [a];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradA = new Float32Array(a.size);
                const contStrides = computeStrides([...a.shape]);
                const idx = new Array(ndim).fill(0);
                for (let i = 0; i < outSize; i++) {
                    let dstOffset = 0;
                    for (let d = 0; d < ndim; d++) {
                        dstOffset += (starts[d] + idx[d]) * contStrides[d];
                    }
                    gradA[dstOffset] += gData[i];
                    incrementIndices(idx, outShape);
                }
                accumGrad(a, gradA);
            };
        }
        return result;
    }
    // ─── Matrix Ops ───────────────────────────────────────────────────────
    function matmul(a, b) {
        const aNdim = a.shape.length;
        const bNdim = b.shape.length;
        assert(aNdim >= 2 && bNdim >= 2, `matmul requires at least 2D tensors`);
        const M = a.shape[aNdim - 2];
        const K = a.shape[aNdim - 1];
        const N = b.shape[bNdim - 1];
        assert(K === b.shape[bNdim - 2], `matmul inner dims must match: ${K} vs ${b.shape[bNdim - 2]}`);
        const aBatch = a.shape.slice(0, -2);
        const bBatch = b.shape.slice(0, -2);
        const batchShape = aBatch.length > 0 || bBatch.length > 0
            ? broadcastShape(aBatch, bBatch)
            : [];
        const outShape = [...batchShape, M, N];
        const batchSize = Math.max(shapeSize(batchShape), 1);
        const aData = a.contiguousData();
        const bData = b.contiguousData();
        const data = new Float32Array(shapeSize(outShape));
        const aMatSize = M * K;
        const bMatSize = K * N;
        const oMatSize = M * N;
        const aBatchSize = shapeSize(aBatch.length > 0 ? aBatch : [1]);
        const bBatchSize = shapeSize(bBatch.length > 0 ? bBatch : [1]);
        for (let batch = 0; batch < batchSize; batch++) {
            const aSlice = aBatchSize === 1 ? 0 : (aBatchSize === batchSize ? batch : batch % aBatchSize);
            const bSlice = bBatchSize === 1 ? 0 : (bBatchSize === batchSize ? batch : batch % bBatchSize);
            const aOffset = aSlice * aMatSize;
            const bOffset = bSlice * bMatSize;
            const oOffset = batch * oMatSize;
            // i-k-j loop order: cache-friendly for both A (row-major) and C (row-major)
            for (let i = 0; i < M; i++) {
                const aRowOff = aOffset + i * K;
                const oRowOff = oOffset + i * N;
                for (let k = 0; k < K; k++) {
                    const aVal = aData[aRowOff + k];
                    const bRowOff = bOffset + k * N;
                    for (let j = 0; j < N; j++) {
                        data[oRowOff + j] += aVal * bData[bRowOff + j];
                    }
                }
            }
        }
        const g = needsGrad(a, b);
        const result = new Tensor(data, outShape, undefined, g);
        if (g) {
            result._children = [a, b].filter((t) => t.requiresGrad);
            result._backward = () => {
                const gData = result.grad.contiguousData();
                if (a.requiresGrad) {
                    if (!a.grad)
                        a.grad = Tensor.zeros([...a.shape]);
                    const aGrad = a.grad.data;
                    for (let batch = 0; batch < batchSize; batch++) {
                        const aSlice = aBatchSize === 1 ? 0 : (aBatchSize === batchSize ? batch : batch % aBatchSize);
                        const bSlice = bBatchSize === 1 ? 0 : (bBatchSize === batchSize ? batch : batch % bBatchSize);
                        const aOff = aSlice * aMatSize;
                        const bOff = bSlice * bMatSize;
                        const gOff = batch * oMatSize;
                        // dL/dA = dL/dC @ B^T
                        for (let i = 0; i < M; i++) {
                            for (let k = 0; k < K; k++) {
                                let s = 0;
                                for (let j = 0; j < N; j++) {
                                    s += gData[gOff + i * N + j] * bData[bOff + k * N + j];
                                }
                                aGrad[aOff + i * K + k] += s;
                            }
                        }
                    }
                }
                if (b.requiresGrad) {
                    if (!b.grad)
                        b.grad = Tensor.zeros([...b.shape]);
                    const bGrad = b.grad.data;
                    for (let batch = 0; batch < batchSize; batch++) {
                        const aSlice = aBatchSize === 1 ? 0 : (aBatchSize === batchSize ? batch : batch % aBatchSize);
                        const bSlice = bBatchSize === 1 ? 0 : (bBatchSize === batchSize ? batch : batch % bBatchSize);
                        const aOff = aSlice * aMatSize;
                        const bOff = bSlice * bMatSize;
                        const gOff = batch * oMatSize;
                        // dL/dB = A^T @ dL/dC
                        for (let k = 0; k < K; k++) {
                            for (let j = 0; j < N; j++) {
                                let s = 0;
                                for (let i = 0; i < M; i++) {
                                    s += aData[aOff + i * K + k] * gData[gOff + i * N + j];
                                }
                                bGrad[bOff + k * N + j] += s;
                            }
                        }
                    }
                }
            };
        }
        return result;
    }
    function linear(x, weight, bias) {
        // Fused x @ weight^T + bias without materializing the transpose.
        // x: [..., inF], weight: [outF, inF] → result: [..., outF]
        const xNdim = x.shape.length;
        assert(xNdim >= 1, 'linear input must be at least 1D');
        assert(weight.shape.length === 2, 'linear weight must be 2D');
        const inF = x.shape[xNdim - 1];
        const outF = weight.shape[0];
        assert(weight.shape[1] === inF, `linear weight shape mismatch: [${weight.shape}] vs input dim ${inF}`);
        const xData = x.contiguousData();
        const wData = weight.contiguousData();
        const bData = bias ? bias.contiguousData() : null;
        // Compute batch dimensions (everything except last dim)
        const batchDims = x.shape.slice(0, -1);
        const batchSize = shapeSize(batchDims.length > 0 ? batchDims : [1]);
        const outShape = [...batchDims, outF];
        const data = new Float32Array(batchSize * outF);
        // x @ W^T: for each (batch, outFeature), dot product over inFeatures.
        // b-j-k order: k inner loop reads contiguous x[b,:] and W[j,:].
        for (let b = 0; b < batchSize; b++) {
            const xOff = b * inF;
            const oOff = b * outF;
            for (let j = 0; j < outF; j++) {
                const wRowOff = j * inF;
                let s = bData ? bData[j] : 0;
                for (let k = 0; k < inF; k++) {
                    s += xData[xOff + k] * wData[wRowOff + k];
                }
                data[oOff + j] = s;
            }
        }
        const g = bias ? needsGrad(x, weight, bias) : needsGrad(x, weight);
        const result = new Tensor(data, outShape, undefined, g);
        if (g) {
            const parents = [x, weight];
            if (bias)
                parents.push(bias);
            result._children = parents.filter((t) => t.requiresGrad);
            result._backward = () => {
                const gData = result.grad.contiguousData();
                if (x.requiresGrad) {
                    // dL/dx = dL/dy @ W (not transposed)
                    const gradX = new Float32Array(x.size);
                    for (let b = 0; b < batchSize; b++) {
                        const gOff = b * outF;
                        const xOff = b * inF;
                        for (let j = 0; j < outF; j++) {
                            const gVal = gData[gOff + j];
                            const wRowOff = j * inF;
                            for (let k = 0; k < inF; k++) {
                                gradX[xOff + k] += gVal * wData[wRowOff + k];
                            }
                        }
                    }
                    accumGrad(x, gradX);
                }
                if (weight.requiresGrad) {
                    // dL/dW[j,k] = sum_b dL/dy[b,j] * x[b,k]
                    // j-b-k order: accumulate one row of gradW at a time
                    const gradW = new Float32Array(weight.size);
                    for (let j = 0; j < outF; j++) {
                        const wRowOff = j * inF;
                        for (let b = 0; b < batchSize; b++) {
                            const gVal = gData[b * outF + j];
                            const xOff = b * inF;
                            for (let k = 0; k < inF; k++) {
                                gradW[wRowOff + k] += gVal * xData[xOff + k];
                            }
                        }
                    }
                    accumGrad(weight, gradW);
                }
                if (bias && bias.requiresGrad) {
                    // dL/db = sum of dL/dy over batch
                    const gradB = new Float32Array(outF);
                    for (let b = 0; b < batchSize; b++) {
                        const gOff = b * outF;
                        for (let j = 0; j < outF; j++) {
                            gradB[j] += gData[gOff + j];
                        }
                    }
                    accumGrad(bias, gradB);
                }
            };
        }
        return result;
    }
    // ─── Activation Ops ───────────────────────────────────────────────────
    function exp(a) {
        const aData = a.contiguousData();
        const data = new Float32Array(a.size);
        for (let i = 0; i < a.size; i++)
            data[i] = Math.exp(aData[i]);
        const result = new Tensor(data, [...a.shape], undefined, a.requiresGrad);
        if (a.requiresGrad) {
            result._children = [a];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradA = new Float32Array(a.size);
                for (let i = 0; i < a.size; i++)
                    gradA[i] = gData[i] * data[i];
                accumGrad(a, gradA);
            };
        }
        return result;
    }
    function softmax(a, dim) {
        const ndim = a.shape.length;
        const axis = dim < 0 ? ndim + dim : dim;
        assert(axis >= 0 && axis < ndim, `dim ${dim} out of range for ${ndim}D tensor`);
        const aData = a.contiguousData();
        const data = new Float32Array(a.size);
        const aStrides = computeStrides([...a.shape]);
        const axisLen = a.shape[axis];
        // Compute outer iteration shape (all dims except axis)
        const outerShape = [];
        for (let d = 0; d < ndim; d++) {
            if (d !== axis)
                outerShape.push(a.shape[d]);
        }
        const outerSize = shapeSize(outerShape.length > 0 ? outerShape : [1]);
        const indices = new Array(ndim).fill(0);
        const outerStrides = computeStrides(outerShape.length > 0 ? outerShape : [1]);
        for (let outer = 0; outer < outerSize; outer++) {
            // Decode outer index into non-axis dimensions
            const outerIdx = [];
            let rem = outer;
            for (let d = 0; d < outerShape.length; d++) {
                outerIdx.push(Math.floor(rem / outerStrides[d]));
                rem = rem % outerStrides[d];
            }
            let oi = 0;
            for (let d = 0; d < ndim; d++) {
                if (d !== axis) {
                    indices[d] = outerIdx[oi];
                    oi++;
                }
            }
            // Find max
            let maxVal = -Infinity;
            for (let k = 0; k < axisLen; k++) {
                indices[axis] = k;
                let offset = 0;
                for (let d = 0; d < ndim; d++)
                    offset += indices[d] * aStrides[d];
                if (aData[offset] > maxVal)
                    maxVal = aData[offset];
            }
            // Compute exp and sum
            let sumExp = 0;
            for (let k = 0; k < axisLen; k++) {
                indices[axis] = k;
                let offset = 0;
                for (let d = 0; d < ndim; d++)
                    offset += indices[d] * aStrides[d];
                const e = Math.exp(aData[offset] - maxVal);
                data[offset] = e;
                sumExp += e;
            }
            // Normalize
            for (let k = 0; k < axisLen; k++) {
                indices[axis] = k;
                let offset = 0;
                for (let d = 0; d < ndim; d++)
                    offset += indices[d] * aStrides[d];
                data[offset] /= sumExp;
            }
        }
        const result = new Tensor(data, [...a.shape], undefined, a.requiresGrad);
        if (a.requiresGrad) {
            result._children = [a];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradA = new Float32Array(a.size);
                for (let outer = 0; outer < outerSize; outer++) {
                    const outerIdx = [];
                    let rem = outer;
                    for (let d = 0; d < outerShape.length; d++) {
                        outerIdx.push(Math.floor(rem / outerStrides[d]));
                        rem = rem % outerStrides[d];
                    }
                    let oi = 0;
                    for (let d = 0; d < ndim; d++) {
                        if (d !== axis) {
                            indices[d] = outerIdx[oi];
                            oi++;
                        }
                    }
                    // dot(grad, softmax)
                    let dot = 0;
                    for (let k = 0; k < axisLen; k++) {
                        indices[axis] = k;
                        let offset = 0;
                        for (let d = 0; d < ndim; d++)
                            offset += indices[d] * aStrides[d];
                        dot += gData[offset] * data[offset];
                    }
                    for (let k = 0; k < axisLen; k++) {
                        indices[axis] = k;
                        let offset = 0;
                        for (let d = 0; d < ndim; d++)
                            offset += indices[d] * aStrides[d];
                        gradA[offset] = data[offset] * (gData[offset] - dot);
                    }
                }
                accumGrad(a, gradA);
            };
        }
        return result;
    }
    function gelu(a) {
        const SQRT_2_PI = Math.sqrt(2 / Math.PI);
        const aData = a.contiguousData();
        const data = new Float32Array(a.size);
        const tanhVals = new Float32Array(a.size);
        for (let i = 0; i < a.size; i++) {
            const x = aData[i];
            const inner = SQRT_2_PI * (x + 0.044715 * x * x * x);
            const t = Math.tanh(inner);
            tanhVals[i] = t;
            data[i] = 0.5 * x * (1 + t);
        }
        const result = new Tensor(data, [...a.shape], undefined, a.requiresGrad);
        if (a.requiresGrad) {
            result._children = [a];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradA = new Float32Array(a.size);
                for (let i = 0; i < a.size; i++) {
                    const x = aData[i];
                    const t = tanhVals[i];
                    const sech2 = 1 - t * t;
                    const innerDeriv = SQRT_2_PI * (1 + 3 * 0.044715 * x * x);
                    gradA[i] = gData[i] * (0.5 * (1 + t) + 0.5 * x * sech2 * innerDeriv);
                }
                accumGrad(a, gradA);
            };
        }
        return result;
    }
    // ─── Normalization ────────────────────────────────────────────────────
    function layerNorm(x, weight, bias, eps = 1e-5) {
        const ndim = x.shape.length;
        const lastDim = x.shape[ndim - 1];
        assert(weight.size === lastDim, `weight size ${weight.size} must match last dim ${lastDim}`);
        assert(bias.size === lastDim, `bias size ${bias.size} must match last dim ${lastDim}`);
        const xData = x.contiguousData();
        const wData = weight.contiguousData();
        const bData = bias.contiguousData();
        const numVectors = x.size / lastDim;
        const data = new Float32Array(x.size);
        const invStds = new Float32Array(numVectors);
        const xNormed = new Float32Array(x.size);
        for (let v = 0; v < numVectors; v++) {
            const offset = v * lastDim;
            let m = 0;
            for (let i = 0; i < lastDim; i++)
                m += xData[offset + i];
            m /= lastDim;
            let variance = 0;
            for (let i = 0; i < lastDim; i++) {
                const diff = xData[offset + i] - m;
                variance += diff * diff;
            }
            variance /= lastDim;
            const invStd = 1 / Math.sqrt(variance + eps);
            invStds[v] = invStd;
            for (let i = 0; i < lastDim; i++) {
                const normed = (xData[offset + i] - m) * invStd;
                xNormed[offset + i] = normed;
                data[offset + i] = normed * wData[i] + bData[i];
            }
        }
        const g = needsGrad(x, weight, bias);
        const result = new Tensor(data, [...x.shape], undefined, g);
        if (g) {
            result._children = [x, weight, bias].filter((t) => t.requiresGrad);
            result._backward = () => {
                const gData = result.grad.contiguousData();
                if (x.requiresGrad) {
                    const gradX = new Float32Array(x.size);
                    for (let v = 0; v < numVectors; v++) {
                        const offset = v * lastDim;
                        const invStd = invStds[v];
                        let dotGW = 0;
                        let dotGX = 0;
                        for (let i = 0; i < lastDim; i++) {
                            const gi = gData[offset + i] * wData[i];
                            dotGW += gi;
                            dotGX += gi * xNormed[offset + i];
                        }
                        for (let i = 0; i < lastDim; i++) {
                            const gi = gData[offset + i] * wData[i];
                            gradX[offset + i] = invStd * (gi - dotGW / lastDim - xNormed[offset + i] * dotGX / lastDim);
                        }
                    }
                    accumGrad(x, gradX);
                }
                if (weight.requiresGrad) {
                    const gradW = new Float32Array(lastDim);
                    for (let v = 0; v < numVectors; v++) {
                        const offset = v * lastDim;
                        for (let i = 0; i < lastDim; i++) {
                            gradW[i] += gData[offset + i] * xNormed[offset + i];
                        }
                    }
                    accumGrad(weight, gradW);
                }
                if (bias.requiresGrad) {
                    const gradB = new Float32Array(lastDim);
                    for (let v = 0; v < numVectors; v++) {
                        const offset = v * lastDim;
                        for (let i = 0; i < lastDim; i++) {
                            gradB[i] += gData[offset + i];
                        }
                    }
                    accumGrad(bias, gradB);
                }
            };
        }
        return result;
    }
    // ─── Loss Functions ───────────────────────────────────────────────────
    function crossEntropy(logits, targets) {
        const ndim = logits.shape.length;
        assert(ndim === 1 || ndim === 2, `crossEntropy expects 1D or 2D logits, got ${ndim}D`);
        const logitsData = logits.contiguousData();
        const batchSize = ndim === 2 ? logits.shape[0] : 1;
        const vocabSize = logits.shape[ndim - 1];
        assert(targets.length === batchSize, `targets length ${targets.length} must match batch size ${batchSize}`);
        const probs = new Float32Array(logits.size);
        let totalLoss = 0;
        for (let b = 0; b < batchSize; b++) {
            const offset = b * vocabSize;
            let maxVal = -Infinity;
            for (let i = 0; i < vocabSize; i++) {
                if (logitsData[offset + i] > maxVal)
                    maxVal = logitsData[offset + i];
            }
            let sumExp = 0;
            for (let i = 0; i < vocabSize; i++) {
                const e = Math.exp(logitsData[offset + i] - maxVal);
                probs[offset + i] = e;
                sumExp += e;
            }
            for (let i = 0; i < vocabSize; i++)
                probs[offset + i] /= sumExp;
            totalLoss -= Math.log(probs[offset + targets[b]] + 1e-10);
        }
        const loss = totalLoss / batchSize;
        const result = new Tensor(new Float32Array([loss]), [], undefined, logits.requiresGrad);
        if (logits.requiresGrad) {
            result._children = [logits];
            result._backward = () => {
                const g = result.grad.data[0];
                const gradLogits = new Float32Array(logits.size);
                for (let b = 0; b < batchSize; b++) {
                    const offset = b * vocabSize;
                    const targetIdx = targets[b];
                    for (let i = 0; i < vocabSize; i++) {
                        gradLogits[offset + i] = g * (probs[offset + i] - (i === targetIdx ? 1 : 0)) / batchSize;
                    }
                }
                accumGrad(logits, gradLogits);
            };
        }
        return result;
    }
    function mse(prediction, target) {
        assert(shapeEqual(prediction.shape, target.shape), `mse shape mismatch: [${prediction.shape}] vs [${target.shape}]`);
        const pData = prediction.contiguousData();
        const tData = target.contiguousData();
        let totalLoss = 0;
        for (let i = 0; i < prediction.size; i++) {
            const diff = pData[i] - tData[i];
            totalLoss += diff * diff;
        }
        const loss = totalLoss / prediction.size;
        const result = new Tensor(new Float32Array([loss]), [], undefined, prediction.requiresGrad);
        if (prediction.requiresGrad) {
            result._children = [prediction];
            result._backward = () => {
                const g = result.grad.data[0];
                const gradP = new Float32Array(prediction.size);
                for (let i = 0; i < prediction.size; i++) {
                    gradP[i] = g * 2 * (pData[i] - tData[i]) / prediction.size;
                }
                accumGrad(prediction, gradP);
            };
        }
        return result;
    }
    // ─── Embedding ────────────────────────────────────────────────────────
    function embedding(indices, table) {
        assert(table.shape.length === 2, `embedding table must be 2D, got ${table.shape.length}D`);
        const vocabSize = table.shape[0];
        const embedDim = table.shape[1];
        const seqLen = indices.length;
        const tableData = table.contiguousData();
        const data = new Float32Array(seqLen * embedDim);
        for (let i = 0; i < seqLen; i++) {
            const idx = indices[i];
            assert(idx >= 0 && idx < vocabSize, `embedding index ${idx} out of range [0, ${vocabSize})`);
            const srcOffset = idx * embedDim;
            for (let j = 0; j < embedDim; j++) {
                data[i * embedDim + j] = tableData[srcOffset + j];
            }
        }
        const outShape = [seqLen, embedDim];
        const result = new Tensor(data, outShape, undefined, table.requiresGrad);
        if (table.requiresGrad) {
            result._children = [table];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradTable = new Float32Array(table.size);
                for (let i = 0; i < seqLen; i++) {
                    const idx = indices[i];
                    const srcOffset = idx * embedDim;
                    for (let j = 0; j < embedDim; j++) {
                        gradTable[srcOffset + j] += gData[i * embedDim + j];
                    }
                }
                accumGrad(table, gradTable);
            };
        }
        return result;
    }
    // ─── Concatenation ────────────────────────────────────────────────────
    function cat(tensors, dim) {
        assert(tensors.length > 0, 'cat requires at least one tensor');
        const ndim = tensors[0].shape.length;
        const axis = dim < 0 ? ndim + dim : dim;
        assert(axis >= 0 && axis < ndim, `dim ${dim} out of range for ${ndim}D tensor`);
        // Validate shapes match on all dims except axis
        for (let t = 1; t < tensors.length; t++) {
            assert(tensors[t].shape.length === ndim, `All tensors must have same number of dimensions`);
            for (let d = 0; d < ndim; d++) {
                if (d !== axis) {
                    assert(tensors[t].shape[d] === tensors[0].shape[d], `Shape mismatch on dim ${d}: ${tensors[t].shape[d]} vs ${tensors[0].shape[d]}`);
                }
            }
        }
        // Compute output shape
        const outShape = [...tensors[0].shape];
        let totalAxisLen = 0;
        const axisOffsets = []; // starting offset for each tensor along axis
        for (const t of tensors) {
            axisOffsets.push(totalAxisLen);
            totalAxisLen += t.shape[axis];
        }
        outShape[axis] = totalAxisLen;
        const outSize = shapeSize(outShape);
        const data = new Float32Array(outSize);
        const outStrides = computeStrides(outShape);
        // Copy each tensor's data into the output
        for (let ti = 0; ti < tensors.length; ti++) {
            const t = tensors[ti];
            const tData = t.contiguousData();
            const tStrides = computeStrides([...t.shape]);
            const tSize = t.size;
            const axisOff = axisOffsets[ti];
            const srcIdx = new Array(ndim).fill(0);
            for (let i = 0; i < tSize; i++) {
                // Compute output index: same as source but shifted on axis
                let outOffset = 0;
                for (let d = 0; d < ndim; d++) {
                    const idx = d === axis ? srcIdx[d] + axisOff : srcIdx[d];
                    outOffset += idx * outStrides[d];
                }
                let srcOffset = 0;
                for (let d = 0; d < ndim; d++) {
                    srcOffset += srcIdx[d] * tStrides[d];
                }
                data[outOffset] = tData[srcOffset];
                incrementIndices(srcIdx, [...t.shape]);
            }
        }
        const g = tensors.some((t) => t.requiresGrad);
        const result = new Tensor(data, outShape, undefined, g);
        if (g) {
            result._children = tensors.filter((t) => t.requiresGrad);
            result._backward = () => {
                const gData = result.grad.contiguousData();
                for (let ti = 0; ti < tensors.length; ti++) {
                    const t = tensors[ti];
                    if (!t.requiresGrad)
                        continue;
                    const tSize = t.size;
                    const tStrides = computeStrides([...t.shape]);
                    const axisOff = axisOffsets[ti];
                    const gradT = new Float32Array(tSize);
                    const srcIdx = new Array(ndim).fill(0);
                    for (let i = 0; i < tSize; i++) {
                        let outOffset = 0;
                        for (let d = 0; d < ndim; d++) {
                            const idx = d === axis ? srcIdx[d] + axisOff : srcIdx[d];
                            outOffset += idx * outStrides[d];
                        }
                        let srcOffset = 0;
                        for (let d = 0; d < ndim; d++) {
                            srcOffset += srcIdx[d] * tStrides[d];
                        }
                        gradT[srcOffset] = gData[outOffset];
                        incrementIndices(srcIdx, [...t.shape]);
                    }
                    accumGrad(t, gradT);
                }
            };
        }
        return result;
    }
    // ─── Rotary Position Embeddings ───────────────────────────────────────
    function rope(x, startPos = 0, rotaryDim) {
        // x shape: [seqLen, numHeads, headDim]
        const ndim = x.shape.length;
        assert(ndim === 3, `rope expects 3D tensor [seqLen, numHeads, headDim], got ${ndim}D`);
        const seqLen = x.shape[0];
        const numHeads = x.shape[1];
        const headDim = x.shape[2];
        // rotaryDim: how many dimensions of headDim to rotate (default: all)
        const rDim = rotaryDim ?? headDim;
        assert(rDim % 2 === 0, `rotaryDim must be even, got ${rDim}`);
        assert(rDim <= headDim, `rotaryDim ${rDim} must be <= headDim ${headDim}`);
        const xData = x.contiguousData();
        const data = new Float32Array(x.size);
        // Copy all data first (non-rotated dims pass through unchanged)
        data.set(xData);
        // Precompute cos/sin for each position and frequency pair
        const halfRDim = rDim / 2;
        const cosVals = new Float32Array(seqLen * halfRDim);
        const sinVals = new Float32Array(seqLen * halfRDim);
        for (let pos = 0; pos < seqLen; pos++) {
            for (let i = 0; i < halfRDim; i++) {
                const freq = 1.0 / Math.pow(10000, (2 * i) / rDim);
                const angle = (pos + startPos) * freq;
                cosVals[pos * halfRDim + i] = Math.cos(angle);
                sinVals[pos * halfRDim + i] = Math.sin(angle);
            }
        }
        // Apply rotation only to the first rotaryDim dimensions
        for (let pos = 0; pos < seqLen; pos++) {
            for (let h = 0; h < numHeads; h++) {
                const baseOffset = (pos * numHeads + h) * headDim;
                for (let i = 0; i < halfRDim; i++) {
                    const cos = cosVals[pos * halfRDim + i];
                    const sin = sinVals[pos * halfRDim + i];
                    const x0 = xData[baseOffset + 2 * i];
                    const x1 = xData[baseOffset + 2 * i + 1];
                    data[baseOffset + 2 * i] = x0 * cos - x1 * sin;
                    data[baseOffset + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }
        const result = new Tensor(data, [...x.shape], undefined, x.requiresGrad);
        if (x.requiresGrad) {
            result._children = [x];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradX = new Float32Array(x.size);
                // Copy all gradients first (non-rotated dims pass through)
                gradX.set(gData);
                // Inverse rotation on the rotated dimensions
                for (let pos = 0; pos < seqLen; pos++) {
                    for (let h = 0; h < numHeads; h++) {
                        const baseOffset = (pos * numHeads + h) * headDim;
                        for (let i = 0; i < halfRDim; i++) {
                            const cos = cosVals[pos * halfRDim + i];
                            const sin = sinVals[pos * halfRDim + i];
                            const g0 = gData[baseOffset + 2 * i];
                            const g1 = gData[baseOffset + 2 * i + 1];
                            gradX[baseOffset + 2 * i] = g0 * cos + g1 * sin;
                            gradX[baseOffset + 2 * i + 1] = -g0 * sin + g1 * cos;
                        }
                    }
                }
                accumGrad(x, gradX);
            };
        }
        return result;
    }
    // ─── Causal Mask ──────────────────────────────────────────────────────
    function causalMask(seqLen) {
        const data = new Float32Array(seqLen * seqLen);
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                data[i * seqLen + j] = j <= i ? 0 : -Infinity;
            }
        }
        return new Tensor(data, [seqLen, seqLen]);
    }
    // ─── Wire convenience methods onto Tensor prototype ───────────────────
    Tensor.prototype.add = function (other) {
        return add(this, other);
    };
    Tensor.prototype.mul = function (other) {
        return mul(this, other);
    };
    Tensor.prototype.matmul = function (other) {
        return matmul(this, other);
    };
    Tensor.prototype.reshape = function (shape) {
        return reshape(this, shape);
    };
    Tensor.prototype.transpose = function (dim0, dim1) {
        return transpose(this, dim0, dim1);
    };
    Tensor.prototype.sum = function (dim, keepdim) {
        return sum(this, dim, keepdim);
    };
    Tensor.prototype.mean = function (dim, keepdim) {
        return mean(this, dim, keepdim);
    };
    Tensor.prototype.neg = function () {
        return neg(this);
    };

    var ops = /*#__PURE__*/Object.freeze({
        __proto__: null,
        add: add,
        cat: cat,
        causalMask: causalMask,
        crossEntropy: crossEntropy,
        div: div,
        embedding: embedding,
        exp: exp,
        gelu: gelu,
        layerNorm: layerNorm,
        linear: linear,
        matmul: matmul,
        max: max,
        mean: mean,
        mse: mse,
        mul: mul,
        neg: neg,
        reshape: reshape,
        rope: rope,
        slice: slice,
        softmax: softmax,
        sub: sub,
        sum: sum,
        transpose: transpose
    });

    class SGD {
        _params;
        _lr;
        _weightDecay;
        constructor(params, lr, weightDecay = 0) {
            this._params = params;
            this._lr = lr;
            this._weightDecay = weightDecay;
        }
        step() {
            for (const param of this._params) {
                if (!param.grad)
                    continue;
                const gradData = param.grad.contiguousData();
                const data = param.data;
                for (let i = 0; i < data.length; i++) {
                    let g = gradData[i];
                    if (this._weightDecay > 0) {
                        g += this._weightDecay * data[i];
                    }
                    data[i] -= this._lr * g;
                }
            }
        }
        zeroGrad() {
            for (const param of this._params) {
                param.grad = null;
            }
        }
    }
    class Adam {
        _params;
        _lr;
        _beta1;
        _beta2;
        _eps;
        _weightDecay;
        _t;
        _m;
        _v;
        constructor(params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0) {
            this._params = params;
            this._lr = lr;
            this._beta1 = beta1;
            this._beta2 = beta2;
            this._eps = eps;
            this._weightDecay = weightDecay;
            this._t = 0;
            this._m = params.map((p) => new Float32Array(p.size));
            this._v = params.map((p) => new Float32Array(p.size));
        }
        step() {
            this._t++;
            const bc1 = 1 - Math.pow(this._beta1, this._t);
            const bc2 = 1 - Math.pow(this._beta2, this._t);
            for (let pi = 0; pi < this._params.length; pi++) {
                const param = this._params[pi];
                if (!param.grad)
                    continue;
                const gradData = param.grad.contiguousData();
                const data = param.data;
                const m = this._m[pi];
                const v = this._v[pi];
                for (let i = 0; i < data.length; i++) {
                    const g = gradData[i];
                    // Update moments
                    m[i] = this._beta1 * m[i] + (1 - this._beta1) * g;
                    v[i] = this._beta2 * v[i] + (1 - this._beta2) * g * g;
                    // Bias-corrected moments
                    const mHat = m[i] / bc1;
                    const vHat = v[i] / bc2;
                    // AdamW: decoupled weight decay
                    if (this._weightDecay > 0) {
                        data[i] -= this._lr * this._weightDecay * data[i];
                    }
                    data[i] -= this._lr * mHat / (Math.sqrt(vHat) + this._eps);
                }
            }
        }
        zeroGrad() {
            for (const param of this._params) {
                param.grad = null;
            }
        }
        /** Expose moments for testing. */
        get moments() {
            return { m: this._m, v: this._v, t: this._t };
        }
    }

    /**
     * Byte-level BPE tokenizer compatible with HuggingFace tokenizer.json format.
     * Implements GPT-NeoX / Pythia tokenization.
     */
    class Tokenizer {
        _vocab;
        _decoder;
        _mergeRanks;
        _addedTokens;
        _addedTokensDecoder;
        _byteEncoder;
        _byteDecoder;
        _preTokenizeRegex;
        constructor(vocab, mergeRanks, addedTokens) {
            this._vocab = vocab;
            this._mergeRanks = mergeRanks;
            this._addedTokens = addedTokens;
            // Build decoder (id -> token)
            this._decoder = new Map();
            for (const [token, id] of vocab) {
                this._decoder.set(id, token);
            }
            this._addedTokensDecoder = new Map();
            for (const [token, id] of addedTokens) {
                this._addedTokensDecoder.set(id, token);
            }
            // Byte-level encoding table
            this._byteEncoder = buildByteEncoder();
            this._byteDecoder = new Map();
            for (const [byte, char] of this._byteEncoder) {
                this._byteDecoder.set(char, byte);
            }
            // GPT-NeoX pre-tokenization regex
            this._preTokenizeRegex =
                /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
        }
        /**
         * Load tokenizer from a HuggingFace tokenizer.json URL or object.
         */
        static fromJSON(json) {
            const model = json['model'];
            assert(model != null, 'tokenizer.json must have a "model" field');
            // Parse vocab
            const vocabObj = model['vocab'];
            assert(vocabObj != null, 'model.vocab is required');
            const vocab = new Map();
            for (const [token, id] of Object.entries(vocabObj)) {
                vocab.set(token, id);
            }
            // Parse merges — can be string[] ("Ġ t") or string[][] (["Ġ", "t"])
            const mergesArr = model['merges'];
            assert(mergesArr != null, 'model.merges is required');
            const mergeRanks = new Map();
            for (let i = 0; i < mergesArr.length; i++) {
                const merge = mergesArr[i];
                const key = Array.isArray(merge) ? merge.join(' ') : merge;
                mergeRanks.set(key, i);
            }
            // Parse added tokens
            const addedTokens = new Map();
            const addedArr = json['added_tokens'];
            if (addedArr) {
                for (const entry of addedArr) {
                    const content = entry['content'];
                    const id = entry['id'];
                    if (content && typeof id === 'number') {
                        addedTokens.set(content, id);
                    }
                }
            }
            return new Tokenizer(vocab, mergeRanks, addedTokens);
        }
        /**
         * Encode text to token IDs.
         */
        encode(text) {
            if (text.length === 0)
                return [];
            const ids = [];
            // Check for added tokens first (e.g., <|endoftext|>)
            // Simple approach: scan for exact matches of added tokens
            let remaining = text;
            while (remaining.length > 0) {
                let foundAdded = false;
                for (const [token, id] of this._addedTokens) {
                    if (remaining.startsWith(token)) {
                        ids.push(id);
                        remaining = remaining.slice(token.length);
                        foundAdded = true;
                        break;
                    }
                }
                if (foundAdded)
                    continue;
                // Find the next added token position (or end of string)
                let nextAddedPos = remaining.length;
                for (const [token] of this._addedTokens) {
                    const pos = remaining.indexOf(token);
                    if (pos > 0 && pos < nextAddedPos) {
                        nextAddedPos = pos;
                    }
                }
                // Process the chunk before the next added token
                const chunk = remaining.slice(0, nextAddedPos);
                remaining = remaining.slice(nextAddedPos);
                // Pre-tokenize
                const matches = chunk.match(this._preTokenizeRegex);
                if (!matches)
                    continue;
                for (const preToken of matches) {
                    // Convert to byte-level representation
                    const byteStr = this._textToBytes(preToken);
                    // Apply BPE
                    const bpeTokens = this._bpe(byteStr);
                    // Map to IDs
                    for (const t of bpeTokens) {
                        const id = this._vocab.get(t);
                        if (id !== undefined) {
                            ids.push(id);
                        }
                        // Unknown tokens are silently dropped (shouldn't happen with byte-level BPE)
                    }
                }
            }
            return ids;
        }
        /**
         * Decode token IDs to text.
         */
        decode(ids) {
            const tokens = [];
            for (const id of ids) {
                const added = this._addedTokensDecoder.get(id);
                if (added !== undefined) {
                    tokens.push(added);
                    continue;
                }
                const token = this._decoder.get(id);
                if (token !== undefined) {
                    tokens.push(token);
                }
            }
            // Convert byte-level tokens back to text
            const byteStr = tokens.join('');
            return this._bytesToText(byteStr);
        }
        get vocabSize() {
            return this._vocab.size;
        }
        idToToken(id) {
            return this._addedTokensDecoder.get(id) ?? this._decoder.get(id) ?? '<unk>';
        }
        tokenToId(token) {
            return this._addedTokens.get(token) ?? this._vocab.get(token) ?? -1;
        }
        /** Convert text string to byte-level encoded string. */
        _textToBytes(text) {
            const bytes = [];
            const encoder = new TextEncoder();
            const encoded = encoder.encode(text);
            for (const byte of encoded) {
                const char = this._byteEncoder.get(byte);
                if (char) {
                    bytes.push(char);
                }
            }
            return bytes.join('');
        }
        /** Convert byte-level encoded string back to text. */
        _bytesToText(byteStr) {
            const bytes = [];
            for (const char of byteStr) {
                const byte = this._byteDecoder.get(char);
                if (byte !== undefined) {
                    bytes.push(byte);
                }
            }
            return new TextDecoder().decode(new Uint8Array(bytes));
        }
        /** Apply BPE merges to a byte-level string. */
        _bpe(token) {
            if (token.length <= 1)
                return [token];
            // Start with individual characters
            let symbols = [...token];
            while (symbols.length > 1) {
                // Find the best (lowest rank) merge pair
                let bestPair = null;
                let bestRank = Infinity;
                let bestIdx = -1;
                for (let i = 0; i < symbols.length - 1; i++) {
                    const pair = `${symbols[i]} ${symbols[i + 1]}`;
                    const rank = this._mergeRanks.get(pair);
                    if (rank !== undefined && rank < bestRank) {
                        bestRank = rank;
                        bestPair = pair;
                        bestIdx = i;
                    }
                }
                if (bestPair === null)
                    break;
                // Apply the merge at ALL positions where this pair appears
                const merged = symbols[bestIdx] + symbols[bestIdx + 1];
                const newSymbols = [];
                let i = 0;
                while (i < symbols.length) {
                    if (i < symbols.length - 1 &&
                        `${symbols[i]} ${symbols[i + 1]}` === bestPair) {
                        newSymbols.push(merged);
                        i += 2;
                    }
                    else {
                        newSymbols.push(symbols[i]);
                        i++;
                    }
                }
                symbols = newSymbols;
            }
            return symbols;
        }
    }
    /**
     * Build the byte-to-unicode encoder mapping used by GPT-2/NeoX byte-level BPE.
     * Maps bytes 0-255 to unique unicode characters, avoiding control characters.
     */
    function buildByteEncoder() {
        const bs = [];
        const cs = [];
        // Printable ASCII ranges
        for (let i = 33; i <= 126; i++) {
            bs.push(i);
            cs.push(i);
        } // '!' to '~'
        for (let i = 161; i <= 172; i++) {
            bs.push(i);
            cs.push(i);
        } // '¡' to '¬'
        for (let i = 174; i <= 255; i++) {
            bs.push(i);
            cs.push(i);
        } // '®' to 'ÿ'
        // Map remaining bytes (control chars, etc.) to higher unicode
        let n = 0;
        for (let b = 0; b < 256; b++) {
            if (!bs.includes(b)) {
                bs.push(b);
                cs.push(256 + n);
                n++;
            }
        }
        const result = new Map();
        for (let i = 0; i < bs.length; i++) {
            result.set(bs[i], String.fromCharCode(cs[i]));
        }
        return result;
    }

    // ─── URL Resolution ───────────────────────────────────────────────────
    const MODEL_URLS = {
        'pythia-14m': 'https://huggingface.co/tinyllms/pythia-14m/resolve/main/model.json',
        'pythia-70m': 'https://huggingface.co/tinyllms/pythia-70m/resolve/main/model.json',
    };
    const TOKENIZER_URLS = {
        'pythia-14m': 'https://huggingface.co/tinyllms/pythia-14m/resolve/main/tokenizer.json',
        'pythia-70m': 'https://huggingface.co/tinyllms/pythia-70m/resolve/main/tokenizer.json',
    };
    function resolveUrl(urlOrName, urlMap) {
        if (urlOrName.startsWith('http://') || urlOrName.startsWith('https://')) {
            return urlOrName;
        }
        const known = urlMap[urlOrName];
        if (!known)
            throw new Error(`[tinyllms] Unknown model name: ${urlOrName}`);
        return known;
    }
    function baseUrl(manifestUrl) {
        const idx = manifestUrl.lastIndexOf('/');
        return idx >= 0 ? manifestUrl.slice(0, idx + 1) : '';
    }
    // ─── Weight Loading ───────────────────────────────────────────────────
    async function loadWeights(url, options) {
        const manifestUrl = resolveUrl(url, MODEL_URLS);
        // Fetch manifest
        const manifestResp = await fetch(manifestUrl);
        assert(manifestResp.ok, `Failed to fetch manifest: ${manifestResp.status} ${manifestUrl}`);
        const manifest = (await manifestResp.json());
        // Determine unique .bin files and total size
        const fileSet = new Map(); // file -> max(offset + length)
        let totalBytes = 0;
        for (const w of manifest.weights) {
            const current = fileSet.get(w.file) ?? 0;
            const end = w.offset + w.length;
            if (end > current)
                fileSet.set(w.file, end);
            totalBytes += w.length;
        }
        const base = baseUrl(manifestUrl);
        let loadedBytes = 0;
        // Fetch all .bin files in parallel
        const buffers = new Map();
        const fetches = Array.from(fileSet.keys()).map(async (file) => {
            const fileUrl = `${base}${file}`;
            const resp = await fetch(fileUrl);
            assert(resp.ok, `Failed to fetch weights: ${resp.status} ${fileUrl}`);
            if (options?.onProgress && resp.body) {
                // Stream for progress reporting
                const reader = resp.body.getReader();
                const chunks = [];
                while (true) {
                    const { done, value } = await reader.read();
                    if (done)
                        break;
                    chunks.push(value);
                    loadedBytes += value.byteLength;
                    options.onProgress(loadedBytes, totalBytes);
                }
                // Combine chunks
                const totalLen = chunks.reduce((sum, c) => sum + c.byteLength, 0);
                const combined = new Uint8Array(totalLen);
                let offset = 0;
                for (const chunk of chunks) {
                    combined.set(chunk, offset);
                    offset += chunk.byteLength;
                }
                buffers.set(file, combined.buffer);
            }
            else {
                const buf = await resp.arrayBuffer();
                loadedBytes += buf.byteLength;
                if (options?.onProgress)
                    options.onProgress(loadedBytes, totalBytes);
                buffers.set(file, buf);
            }
        });
        await Promise.all(fetches);
        // Create Tensor objects from weight entries
        const weights = new Map();
        for (const w of manifest.weights) {
            const buf = buffers.get(w.file);
            if (!buf)
                throw new Error(`[tinyllms] Missing buffer for file: ${w.file}`);
            const floats = new Float32Array(buf, w.offset, w.length / 4);
            // Copy to own buffer (the slice might share the underlying ArrayBuffer)
            const data = new Float32Array(floats);
            weights.set(w.name, new Tensor(data, w.shape));
        }
        return { manifest, weights };
    }
    // ─── High-level API ───────────────────────────────────────────────────
    async function loadTokenizer(urlOrName) {
        const url = resolveUrl(urlOrName, TOKENIZER_URLS);
        const resp = await fetch(url);
        assert(resp.ok, `Failed to fetch tokenizer: ${resp.status} ${url}`);
        const json = await resp.json();
        return Tokenizer.fromJSON(json);
    }

    /**
     * Registry for forward-pass activation hooks.
     * Hooks are callbacks fired during model forward passes
     * to capture intermediate tensors (attention weights, hidden states, etc.).
     */
    class HookRegistry {
        _hooks = new Map();
        /**
         * Register a hook callback for a named activation point.
         * @param pattern - Exact name (e.g., 'layers.0.attention_weights') or '*' for all.
         * @param callback - Called with (name, tensor) when the activation fires.
         * @returns A function that removes this hook when called.
         */
        add(pattern, callback) {
            if (!this._hooks.has(pattern)) {
                this._hooks.set(pattern, []);
            }
            this._hooks.get(pattern).push(callback);
            return () => {
                const hooks = this._hooks.get(pattern);
                if (hooks) {
                    const idx = hooks.indexOf(callback);
                    if (idx >= 0)
                        hooks.splice(idx, 1);
                    if (hooks.length === 0)
                        this._hooks.delete(pattern);
                }
            };
        }
        /**
         * Fire all hooks matching the given activation name.
         */
        fire(name, tensor) {
            // Exact match
            const exact = this._hooks.get(name);
            if (exact) {
                for (const cb of exact)
                    cb(name, tensor);
            }
            // Wildcard
            const wild = this._hooks.get('*');
            if (wild) {
                for (const cb of wild)
                    cb(name, tensor);
            }
        }
        /** Remove all registered hooks. */
        clear() {
            this._hooks.clear();
        }
        /** True if any hooks are registered. */
        get hasHooks() {
            return this._hooks.size > 0;
        }
    }

    /**
     * WebGPU acceleration backend.
     *
     * Provides GPU-accelerated matmul via a tiled compute shader.
     * Auto-detects WebGPU availability. Falls back gracefully when unavailable.
     */
    // ─── State ────────────────────────────────────────────────────────────
    let _device = null;
    let _initPromise = null;
    let _available = false;
    let _matmulPipeline = null;
    /** Minimum total elements to justify GPU dispatch. */
    const MATMUL_THRESHOLD = 4096;
    const TILE_SIZE = 16;
    // ─── Initialization ───────────────────────────────────────────────────
    /**
     * Initialize the WebGPU backend. Safe to call multiple times.
     * Returns true if WebGPU is available and device was obtained.
     */
    async function initWebGPU() {
        if (_initPromise)
            return _initPromise;
        _initPromise = _doInit();
        return _initPromise;
    }
    async function _doInit() {
        try {
            if (typeof navigator === 'undefined' || !navigator.gpu)
                return false;
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter)
                return false;
            _device = await adapter.requestDevice();
            _device.lost.then(() => {
                _device = null;
                _available = false;
                _matmulPipeline = null;
                _initPromise = null;
            });
            _matmulPipeline = _createMatmulPipeline(_device);
            _available = true;
            return true;
        }
        catch {
            return false;
        }
    }
    function isWebGPUAvailable() {
        return _available && _device !== null;
    }
    function getBackend() {
        return _available ? 'webgpu' : 'cpu';
    }
    // ─── Matmul Shader ────────────────────────────────────────────────────
    const MATMUL_SHADER = /* wgsl */ `
struct Dims {
  M: u32,
  K: u32,
  N: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE: u32 = ${TILE_SIZE}u;

var<workgroup> tileA: array<f32, ${TILE_SIZE * TILE_SIZE}>;
var<workgroup> tileB: array<f32, ${TILE_SIZE * TILE_SIZE}>;

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;
  let M = dims.M;
  let K = dims.K;
  let N = dims.N;

  var acc: f32 = 0.0;
  let numTiles = (K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    let aCol = t * TILE + localCol;
    if (row < M && aCol < K) {
      tileA[localRow * TILE + localCol] = A[row * K + aCol];
    } else {
      tileA[localRow * TILE + localCol] = 0.0;
    }

    let bRow = t * TILE + localRow;
    if (bRow < K && col < N) {
      tileB[localRow * TILE + localCol] = B[bRow * N + col];
    } else {
      tileB[localRow * TILE + localCol] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      acc = acc + tileA[localRow * TILE + k] * tileB[k * TILE + localCol];
    }

    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}
`;
    function _createMatmulPipeline(device) {
        const module = device.createShaderModule({ code: MATMUL_SHADER });
        return device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
    }
    // ─── GPU Matmul ───────────────────────────────────────────────────────
    /**
     * Returns true if this matmul should use the GPU path.
     */
    function shouldUseGPU(M, K, N) {
        if (!_available || !_device || !_matmulPipeline)
            return false;
        return (M * K + K * N) >= MATMUL_THRESHOLD;
    }
    /**
     * GPU-accelerated matrix multiplication: C = A @ B
     * A: [M, K], B: [K, N] → C: [M, N]
     *
     * Async because WebGPU readback requires mapAsync.
     */
    async function gpuMatmul(aData, bData, M, K, N) {
        const device = _device;
        const pipeline = _matmulPipeline;
        const dimsBuf = device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(dimsBuf, 0, new Uint32Array([M, K, N]));
        const aBuf = device.createBuffer({
            size: aData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(aBuf, 0, aData);
        const bBuf = device.createBuffer({
            size: bData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bBuf, 0, bData);
        const cSize = M * N * 4;
        const cBuf = device.createBuffer({
            size: cSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readBuf = device.createBuffer({
            size: cSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: dimsBuf } },
                { binding: 1, resource: { buffer: aBuf } },
                { binding: 2, resource: { buffer: bBuf } },
                { binding: 3, resource: { buffer: cBuf } },
            ],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(N / TILE_SIZE), Math.ceil(M / TILE_SIZE));
        pass.end();
        encoder.copyBufferToBuffer(cBuf, 0, readBuf, 0, cSize);
        device.queue.submit([encoder.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();
        dimsBuf.destroy();
        aBuf.destroy();
        bBuf.destroy();
        cBuf.destroy();
        readBuf.destroy();
        return result;
    }

    // ─── GPU-accelerated linear ───────────────────────────────────────
    /**
     * Async linear projection using GPU matmul when beneficial.
     * Computes x @ weight^T + bias. No autograd (inference only).
     */
    async function linearGPU(x, weight, bias) {
        // x: [seqLen, inFeatures], weight: [outFeatures, inFeatures]
        const seqLen = x.shape[0];
        const inFeatures = x.shape[x.shape.length - 1];
        const outFeatures = weight.shape[0];
        // x @ weight^T: [seqLen, inFeatures] @ [inFeatures, outFeatures]
        if (shouldUseGPU(seqLen, inFeatures, outFeatures)) {
            const xData = x.contiguousData();
            const wData = weight.contiguousData();
            // Need weight transposed: [outFeatures, inFeatures] → [inFeatures, outFeatures]
            const wT = new Float32Array(inFeatures * outFeatures);
            for (let i = 0; i < outFeatures; i++) {
                for (let j = 0; j < inFeatures; j++) {
                    wT[j * outFeatures + i] = wData[i * inFeatures + j];
                }
            }
            const resultData = await gpuMatmul(xData, wT, seqLen, inFeatures, outFeatures);
            // Add bias
            if (bias) {
                const bData = bias.contiguousData();
                for (let i = 0; i < seqLen; i++) {
                    for (let j = 0; j < outFeatures; j++) {
                        resultData[i * outFeatures + j] += bData[j];
                    }
                }
            }
            return new Tensor(resultData, [seqLen, outFeatures]);
        }
        // Fallback to CPU
        return linear(x, weight, bias);
    }
    // ─── Pythia Model ─────────────────────────────────────────────────────
    class PythiaModel {
        config;
        hooks;
        _weights;
        constructor(config, weights) {
            this.config = config;
            this._weights = weights;
            this.hooks = new HookRegistry();
        }
        /** Get a weight tensor by name. Throws if not found. */
        _w(name) {
            const t = this._weights.get(name);
            if (!t)
                throw new Error(`[tinyllms] Weight not found: ${name}`);
            return t;
        }
        forward(tokenIds) {
            const { numLayers } = this.config;
            const seqLen = tokenIds.length;
            // 1. Token embedding (no positional embedding — RoPE handles position)
            let hidden = embedding(tokenIds, this._w('embed.weight'));
            // hidden: [seqLen, hiddenDim]
            if (this.hooks.hasHooks)
                this.hooks.fire('embedding', hidden);
            // 2. Transformer blocks
            for (let i = 0; i < numLayers; i++) {
                hidden = this._transformerBlock(hidden, i, seqLen);
            }
            // 3. Final layer norm
            hidden = layerNorm(hidden, this._w('final_ln.weight'), this._w('final_ln.bias'));
            // 4. LM head
            const logits = linear(hidden, this._w('lm_head.weight'));
            // logits: [seqLen, vocabSize]
            if (this.hooks.hasHooks)
                this.hooks.fire('logits', logits);
            return { logits };
        }
        _transformerBlock(hidden, layer, seqLen) {
            const prefix = `layers.${layer}`;
            // Parallel architecture: attention and MLP use DIFFERENT layernorms
            const attnInput = layerNorm(hidden, this._w(`${prefix}.input_ln.weight`), this._w(`${prefix}.input_ln.bias`));
            const mlpInput = layerNorm(hidden, this._w(`${prefix}.post_attn_ln.weight`), this._w(`${prefix}.post_attn_ln.bias`));
            if (this.hooks.hasHooks)
                this.hooks.fire(`${prefix}.ln_out`, attnInput);
            // Parallel: attention and MLP computed independently, then summed
            const attnOut = this._attention(attnInput, layer, seqLen);
            const mlpOut = this._mlp(mlpInput, layer);
            // Residual: hidden + attnOut + mlpOut
            const result = add(add(hidden, attnOut), mlpOut);
            if (this.hooks.hasHooks)
                this.hooks.fire(`${prefix}.output`, result);
            return result;
        }
        _attention(x, layer, seqLen) {
            const { numHeads, headDim, hiddenDim } = this.config;
            const prefix = `layers.${layer}.attention`;
            // QKV projection: [seqLen, hiddenDim] -> [seqLen, 3 * hiddenDim]
            const qkv = linear(x, this._w(`${prefix}.qkv.weight`), this._w(`${prefix}.qkv.bias`));
            // Split into Q, K, V — each [seqLen, hiddenDim]
            // Then reshape to [seqLen, numHeads, headDim]
            const q = reshape(slice(qkv, [0, 0], [seqLen, hiddenDim]), [seqLen, numHeads, headDim]);
            const k = reshape(slice(qkv, [0, hiddenDim], [seqLen, 2 * hiddenDim]), [seqLen, numHeads, headDim]);
            const v = reshape(slice(qkv, [0, 2 * hiddenDim], [seqLen, 3 * hiddenDim]), [seqLen, numHeads, headDim]);
            // Apply RoPE to Q and K (only first rotaryDim dimensions)
            const rotaryDim = this.config.rotaryDim;
            const qRoped = rope(q, 0, rotaryDim);
            const kRoped = rope(k, 0, rotaryDim);
            // Transpose to [numHeads, seqLen, headDim] for batched matmul
            const qT = transpose(qRoped, 0, 1);
            const kT = transpose(kRoped, 0, 1);
            const vT = transpose(v, 0, 1);
            // Attention scores: Q @ K^T / sqrt(headDim)
            const kTT = transpose(kT, 1, 2); // [numHeads, headDim, seqLen]
            const scores = matmul(qT, kTT); // [numHeads, seqLen, seqLen]
            const scale = Tensor.scalar(1 / Math.sqrt(headDim));
            const scaledScores = mul(scores, scale);
            // Causal mask
            const mask = causalMask(seqLen); // [seqLen, seqLen]
            const maskedScores = add(scaledScores, mask); // broadcasts over numHeads
            // Softmax along last dim
            const attnWeights = softmax(maskedScores, -1); // [numHeads, seqLen, seqLen]
            if (this.hooks.hasHooks) {
                this.hooks.fire(`layers.${layer}.attention_weights`, attnWeights);
            }
            // Attention output: weights @ V
            const attnOut = matmul(attnWeights, vT); // [numHeads, seqLen, headDim]
            // Transpose back to [seqLen, numHeads, headDim] then reshape to [seqLen, hiddenDim]
            const attnOutT = transpose(attnOut, 0, 1);
            const concatenated = reshape(attnOutT, [seqLen, hiddenDim]);
            // Output projection
            return linear(concatenated, this._w(`${prefix}.out.weight`), this._w(`${prefix}.out.bias`));
        }
        _mlp(x, layer) {
            const prefix = `layers.${layer}.mlp`;
            // Up projection: hiddenDim -> intermediateSize
            let h = linear(x, this._w(`${prefix}.dense_h_to_4h.weight`), this._w(`${prefix}.dense_h_to_4h.bias`));
            // GELU activation
            h = gelu(h);
            // Down projection: intermediateSize -> hiddenDim
            return linear(h, this._w(`${prefix}.dense_4h_to_h.weight`), this._w(`${prefix}.dense_4h_to_h.bias`));
        }
        // ─── GPU-Accelerated Forward Pass ─────────────────────────────
        /**
         * Async forward pass that uses WebGPU for matmuls when available.
         * Falls back to CPU when WebGPU is not initialized.
         * Does NOT build autograd graph (inference only).
         */
        async forwardGPU(tokenIds) {
            if (!isWebGPUAvailable())
                return this.forward(tokenIds);
            const { numLayers } = this.config;
            const seqLen = tokenIds.length;
            let hidden = embedding(tokenIds, this._w('embed.weight'));
            if (this.hooks.hasHooks)
                this.hooks.fire('embedding', hidden);
            for (let i = 0; i < numLayers; i++) {
                hidden = await this._transformerBlockGPU(hidden, i, seqLen);
            }
            hidden = layerNorm(hidden, this._w('final_ln.weight'), this._w('final_ln.bias'));
            const logits = await linearGPU(hidden, this._w('lm_head.weight'));
            if (this.hooks.hasHooks)
                this.hooks.fire('logits', logits);
            return { logits };
        }
        async _transformerBlockGPU(hidden, layer, seqLen) {
            const prefix = `layers.${layer}`;
            const attnInput = layerNorm(hidden, this._w(`${prefix}.input_ln.weight`), this._w(`${prefix}.input_ln.bias`));
            const mlpInput = layerNorm(hidden, this._w(`${prefix}.post_attn_ln.weight`), this._w(`${prefix}.post_attn_ln.bias`));
            if (this.hooks.hasHooks)
                this.hooks.fire(`${prefix}.ln_out`, attnInput);
            const attnOut = await this._attentionGPU(attnInput, layer, seqLen);
            const mlpOut = await this._mlpGPU(mlpInput, layer);
            const result = add(add(hidden, attnOut), mlpOut);
            if (this.hooks.hasHooks)
                this.hooks.fire(`${prefix}.output`, result);
            return result;
        }
        async _attentionGPU(x, layer, seqLen) {
            const { numHeads, headDim, hiddenDim } = this.config;
            const prefix = `layers.${layer}.attention`;
            const qkv = await linearGPU(x, this._w(`${prefix}.qkv.weight`), this._w(`${prefix}.qkv.bias`));
            const q = reshape(slice(qkv, [0, 0], [seqLen, hiddenDim]), [seqLen, numHeads, headDim]);
            const k = reshape(slice(qkv, [0, hiddenDim], [seqLen, 2 * hiddenDim]), [seqLen, numHeads, headDim]);
            const v = reshape(slice(qkv, [0, 2 * hiddenDim], [seqLen, 3 * hiddenDim]), [seqLen, numHeads, headDim]);
            const rotaryDim = this.config.rotaryDim;
            const qRoped = rope(q, 0, rotaryDim);
            const kRoped = rope(k, 0, rotaryDim);
            const qT = transpose(qRoped, 0, 1);
            const kT = transpose(kRoped, 0, 1);
            const vT = transpose(v, 0, 1);
            const kTT = transpose(kT, 1, 2);
            // Attention matmuls are small (seqLen x seqLen), CPU is fine
            const scores = matmul(qT, kTT);
            const scale = Tensor.scalar(1 / Math.sqrt(headDim));
            const scaledScores = mul(scores, scale);
            const mask = causalMask(seqLen);
            const maskedScores = add(scaledScores, mask);
            const attnWeights = softmax(maskedScores, -1);
            if (this.hooks.hasHooks) {
                this.hooks.fire(`layers.${layer}.attention_weights`, attnWeights);
            }
            const attnOut = matmul(attnWeights, vT);
            const attnOutT = transpose(attnOut, 0, 1);
            const concatenated = reshape(attnOutT, [seqLen, hiddenDim]);
            return linearGPU(concatenated, this._w(`${prefix}.out.weight`), this._w(`${prefix}.out.bias`));
        }
        async _mlpGPU(x, layer) {
            const prefix = `layers.${layer}.mlp`;
            let h = await linearGPU(x, this._w(`${prefix}.dense_h_to_4h.weight`), this._w(`${prefix}.dense_h_to_4h.bias`));
            h = gelu(h);
            return linearGPU(h, this._w(`${prefix}.dense_4h_to_h.weight`), this._w(`${prefix}.dense_4h_to_h.bias`));
        }
        // ─── Generation ─────────────────────────────────────────────────
        async *generate(tokenIds, options) {
            const { maxTokens, temperature = 1.0, topK = 0, stopTokens = [], } = options;
            const tokens = [...tokenIds];
            const useGPU = isWebGPUAvailable();
            for (let step = 0; step < maxTokens; step++) {
                const { logits } = useGPU
                    ? await this.forwardGPU(tokens)
                    : this.forward(tokens);
                const lastLogits = reshape(slice(logits, [tokens.length - 1, 0], [tokens.length, this.config.vocabSize]), [this.config.vocabSize]);
                const nextToken = sampleToken(lastLogits, temperature, topK);
                tokens.push(nextToken);
                if (options.onToken)
                    options.onToken(nextToken);
                yield nextToken;
                if (stopTokens.includes(nextToken))
                    break;
                // Yield to event loop
                if (!useGPU && step % 5 === 0) {
                    await new Promise((r) => setTimeout(r, 0));
                }
            }
        }
    }
    // ─── Sampling ─────────────────────────────────────────────────────────
    function sampleToken(logits, temperature, topK) {
        const data = logits.contiguousData();
        const vocabSize = data.length;
        if (temperature === 0) {
            // Greedy: argmax
            let maxIdx = 0;
            for (let i = 1; i < vocabSize; i++) {
                if (data[i] > data[maxIdx])
                    maxIdx = i;
            }
            return maxIdx;
        }
        // Apply temperature
        const scaled = new Float32Array(vocabSize);
        for (let i = 0; i < vocabSize; i++)
            scaled[i] = data[i] / temperature;
        // Top-k filtering
        if (topK > 0 && topK < vocabSize) {
            const sorted = Array.from(scaled).sort((a, b) => b - a);
            const threshold = sorted[topK - 1];
            for (let i = 0; i < vocabSize; i++) {
                if (scaled[i] < threshold)
                    scaled[i] = -Infinity;
            }
        }
        // Softmax
        let maxVal = -Infinity;
        for (let i = 0; i < vocabSize; i++) {
            if (scaled[i] > maxVal)
                maxVal = scaled[i];
        }
        let sumExp = 0;
        const probs = new Float32Array(vocabSize);
        for (let i = 0; i < vocabSize; i++) {
            probs[i] = Math.exp(scaled[i] - maxVal);
            sumExp += probs[i];
        }
        for (let i = 0; i < vocabSize; i++)
            probs[i] /= sumExp;
        // Sample from categorical distribution
        let r = Math.random();
        for (let i = 0; i < vocabSize; i++) {
            r -= probs[i];
            if (r <= 0)
                return i;
        }
        return vocabSize - 1;
    }

    // ─── Initialization ───────────────────────────────────────────────────
    /** Xavier uniform initialization. */
    function xavierUniform(shape) {
        const fanIn = shape.length >= 2 ? shape[1] : shape[0];
        const fanOut = shape[0];
        const limit = Math.sqrt(6 / (fanIn + fanOut));
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            data[i] = (Math.random() * 2 - 1) * limit;
        }
        return new Tensor(data, shape, undefined, true);
    }
    // ─── ValueHead ────────────────────────────────────────────────────────
    /**
     * Maps each token's hidden state to a scalar value estimate.
     * Linear(d_model, 1) → squeeze last dim.
     * Output: [seqLen] (one value per position).
     */
    class ValueHead {
        _weight;
        _bias;
        constructor(hiddenDim) {
            this._weight = xavierUniform([1, hiddenDim]);
            this._bias = new Tensor(new Float32Array(1), [1], undefined, true);
        }
        forward(hiddenStates) {
            // hiddenStates: [seqLen, hiddenDim]
            const out = linear(hiddenStates, this._weight, this._bias);
            // out: [seqLen, 1]
            const seqLen = hiddenStates.shape[0];
            return reshape(out, [seqLen]); // squeeze to [seqLen]
        }
        parameters() {
            return [this._weight, this._bias];
        }
    }
    // ─── ClassifierHead ───────────────────────────────────────────────────
    /**
     * Maps hidden states to class logits.
     * Linear(d_model, n_classes).
     * Output: [seqLen, nClasses].
     */
    class ClassifierHead {
        _weight;
        _bias;
        constructor(hiddenDim, nClasses) {
            this._weight = xavierUniform([nClasses, hiddenDim]);
            this._bias = new Tensor(new Float32Array(nClasses), [nClasses], undefined, true);
        }
        forward(hiddenStates) {
            // hiddenStates: [seqLen, hiddenDim]
            return linear(hiddenStates, this._weight, this._bias);
            // output: [seqLen, nClasses]
        }
        parameters() {
            return [this._weight, this._bias];
        }
    }
    // ─── RewardHead ───────────────────────────────────────────────────────
    /**
     * Maps the last token's hidden state to a scalar reward.
     * Linear(d_model, 1) applied only to the last position.
     * Output: scalar Tensor.
     */
    class RewardHead {
        _weight;
        _bias;
        constructor(hiddenDim) {
            this._weight = xavierUniform([1, hiddenDim]);
            this._bias = new Tensor(new Float32Array(1), [1], undefined, true);
        }
        forward(hiddenStates) {
            // hiddenStates: [seqLen, hiddenDim]
            const seqLen = hiddenStates.shape[0];
            const hiddenDim = hiddenStates.shape[1];
            // Extract last position: [1, hiddenDim]
            const lastHidden = slice(hiddenStates, [seqLen - 1, 0], [seqLen, hiddenDim]);
            // Linear: [1, 1]
            const out = linear(lastHidden, this._weight, this._bias);
            // Squeeze to scalar [1]
            return reshape(out, [1]);
        }
        parameters() {
            return [this._weight, this._bias];
        }
    }

    // ─── Main-thread Training Loop ────────────────────────────────────────
    /**
     * Training loop that runs on the main thread with yielding.
     * Yields to the event loop every step via setTimeout(0).
     *
     * The lossFn receives a batch (array of examples) and must return
     * a scalar loss Tensor connected to the params via the autograd graph.
     */
    async function train(config) {
        const { lossFn, optimizer, data, epochs, batchSize, onStep, onEpoch, } = config;
        assert(data.length > 0, 'Training data must not be empty');
        assert(batchSize > 0, 'Batch size must be positive');
        const lossHistory = [];
        let totalSteps = 0;
        for (let epoch = 0; epoch < epochs; epoch++) {
            // Shuffle data each epoch (Fisher-Yates)
            const shuffled = [...data];
            for (let i = shuffled.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }
            let epochLossSum = 0;
            let epochSteps = 0;
            for (let i = 0; i < shuffled.length; i += batchSize) {
                const batch = shuffled.slice(i, i + batchSize);
                // Forward + loss
                optimizer.zeroGrad();
                const loss = lossFn(batch);
                const lossVal = loss.item();
                // Backward
                loss.backward();
                // Step
                optimizer.step();
                lossHistory.push(lossVal);
                epochLossSum += lossVal;
                epochSteps++;
                totalSteps++;
                if (onStep) {
                    onStep(totalSteps, lossVal, {});
                }
                // Yield to event loop
                await new Promise((r) => setTimeout(r, 0));
            }
            const avgLoss = epochSteps > 0 ? epochLossSum / epochSteps : 0;
            if (onEpoch) {
                onEpoch(epoch + 1, avgLoss);
            }
        }
        return {
            lossHistory,
            finalLoss: lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : 0,
            totalSteps,
        };
    }

    /**
     * In-memory dataset with batching and shuffling.
     * No browser APIs — works in both Node and browser.
     */
    class Dataset {
        _data;
        constructor(data) {
            this._data = data;
        }
        /** Create dataset from a JSON array of objects. */
        static fromJSON(data) {
            assert(Array.isArray(data), 'Dataset.fromJSON expects an array');
            return new Dataset([...data]);
        }
        /** Create dataset from preference pairs (chosen, rejected). */
        static fromPairs(chosen, rejected) {
            assert(chosen.length === rejected.length, 'chosen and rejected must have same length');
            const data = chosen.map((c, i) => ({ chosen: c, rejected: rejected[i] }));
            return new Dataset(data);
        }
        /** Number of examples in the dataset. */
        get length() {
            return this._data.length;
        }
        /** Get example at index. */
        get(index) {
            assert(index >= 0 && index < this._data.length, `Index ${index} out of range [0, ${this._data.length})`);
            return this._data[index];
        }
        /** In-place Fisher-Yates shuffle. Returns this for chaining. */
        shuffle() {
            for (let i = this._data.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [this._data[i], this._data[j]] = [this._data[j], this._data[i]];
            }
            return this;
        }
        /** Iterate over mini-batches of the given size. */
        *batch(size) {
            assert(size > 0, 'Batch size must be positive');
            for (let i = 0; i < this._data.length; i += size) {
                yield this._data.slice(i, i + size);
            }
        }
        /** Return the raw data array (read-only view). */
        toArray() {
            return this._data;
        }
    }

    /**
     * Bradley-Terry preference loss for reward modeling.
     *
     * Given reward scores for chosen and rejected sequences,
     * loss = -log(sigmoid(r_chosen - r_rejected))
     *      = log(1 + exp(r_rejected - r_chosen))
     *
     * @param chosenReward - Scalar reward for the chosen sequence.
     * @param rejectedReward - Scalar reward for the rejected sequence.
     * @returns Scalar loss tensor.
     */
    function bradleyTerry(chosenReward, rejectedReward) {
        assert(chosenReward.size === 1, `chosenReward must be scalar, got size ${chosenReward.size}`);
        assert(rejectedReward.size === 1, `rejectedReward must be scalar, got size ${rejectedReward.size}`);
        // diff = rejected - chosen
        const diff = sub(reshape(rejectedReward, [1]), reshape(chosenReward, [1]));
        // loss = log(1 + exp(diff))
        // For numerical stability when diff is large positive:
        // log(1 + exp(x)) = x + log(1 + exp(-x)) when x > 0
        // But for simplicity and correctness with autograd, just use the direct formula.
        // The exp op handles the forward, and autograd handles the backward.
        const one = Tensor.scalar(1);
        sum(add(reshape(one, [1]), exp(diff)));
        // log(loss_val) — but we don't have a log op yet.
        // Actually, let me implement this differently.
        // loss = log(1 + exp(diff))
        // We need a log op. Let's implement it inline using the softplus formulation.
        // Actually, we can express this as:
        // -log(sigmoid(chosen - rejected))
        // = -log(1 / (1 + exp(rejected - chosen)))
        // = log(1 + exp(rejected - chosen))
        // Since we don't have a log op in core, let's use the numerically stable
        // softplus: softplus(x) = log(1 + exp(x))
        // For the gradient: d/dx softplus(x) = sigmoid(x) = 1/(1+exp(-x))
        // Let me just compute it directly with the formula and manual backward.
        return softplus(diff);
    }
    /**
     * Softplus: log(1 + exp(x)). Numerically stable implementation.
     * For x > 20, approximates as x (avoids exp overflow).
     */
    function softplus(x) {
        const xData = x.contiguousData();
        const data = new Float32Array(x.size);
        for (let i = 0; i < x.size; i++) {
            const v = xData[i];
            if (v > 20) {
                data[i] = v; // log(1 + exp(v)) ≈ v for large v
            }
            else if (v < -20) {
                data[i] = Math.exp(v); // log(1 + exp(v)) ≈ exp(v) for very negative v
            }
            else {
                data[i] = Math.log(1 + Math.exp(v));
            }
        }
        const result = new Tensor(data, [...x.shape], undefined, x.requiresGrad);
        if (x.requiresGrad) {
            result._children = [x];
            result._backward = () => {
                const gData = result.grad.contiguousData();
                const gradX = new Float32Array(x.size);
                for (let i = 0; i < x.size; i++) {
                    // d/dx softplus(x) = sigmoid(x) = 1/(1+exp(-x))
                    const v = xData[i];
                    const sigmoid = 1 / (1 + Math.exp(-v));
                    gradX[i] = gData[i] * sigmoid;
                }
                if (!x.grad) {
                    x.grad = new Tensor(gradX, [...x.shape]);
                }
                else {
                    const existing = x.grad.contiguousData();
                    for (let i = 0; i < existing.length; i++) {
                        existing[i] += gradX[i];
                    }
                    x.grad = new Tensor(existing, [...x.shape]);
                }
            };
        }
        // Return scalar
        if (x.size === 1) {
            return reshape(result, []);
        }
        return result;
    }

    // ─── Value Function Training ──────────────────────────────────────────
    /**
     * Train a value function using Monte Carlo returns with terminal reward.
     *
     * Algorithm per episode:
     * 1. Generate a sequence from the backbone.
     * 2. Compute reward using rewardFn.
     * 3. Compute Monte Carlo returns: G_t = gamma^(T-t) * R (terminal reward only).
     * 4. Forward pass through backbone to get hidden states.
     * 5. Value head predicts V(s_t) for each position.
     * 6. Loss = MSE(V(s_t), G_t).
     * 7. Backward pass. Update value head parameters.
     */
    async function trainValueFunction(config) {
        const { backbone, valueHead, rewardFn, optimizer, numEpisodes, maxSeqLen, gamma = 0.99, promptTokens, temperature = 1.0, onEpisode, } = config;
        assert(promptTokens.length > 0, 'promptTokens must not be empty');
        for (let episode = 0; episode < numEpisodes; episode++) {
            // 1. Generate sequence
            const tokens = [...promptTokens];
            for await (const token of backbone.generate(tokens, {
                maxTokens: maxSeqLen - promptTokens.length,
                temperature,
            })) {
                tokens.push(token);
            }
            // 2. Compute reward
            const reward = rewardFn(tokens);
            // 3. Compute Monte Carlo returns (terminal reward only)
            // G_t = gamma^(T-t) * R where T is the last position
            const seqLen = tokens.length;
            const returns = new Float32Array(seqLen);
            for (let t = 0; t < seqLen; t++) {
                returns[t] = Math.pow(gamma, seqLen - 1 - t) * reward;
            }
            const targetReturns = new Tensor(returns, [seqLen]);
            // 4. Forward pass through backbone to get hidden states
            // We need the hidden states before the LM head.
            // Use hooks to capture the final layer norm output.
            let hiddenStates = null;
            const removeHook = backbone.hooks.add('logits', () => {
                // We actually need the hidden states BEFORE the LM head.
                // The final hidden states are the input to the LM head.
            });
            // Alternative: capture the last layer's output
            const removeHook2 = backbone.hooks.add(`layers.${backbone.config.numLayers - 1}.output`, (_name, tensor) => {
                hiddenStates = tensor;
            });
            backbone.forward(tokens);
            removeHook();
            removeHook2();
            if (!hiddenStates) {
                throw new Error('[tinyllms] Failed to capture hidden states from backbone');
            }
            // 5. Value head predicts V(s_t)
            optimizer.zeroGrad();
            const values = valueHead.forward(hiddenStates);
            // values: [seqLen]
            // 6. Loss = MSE(values, returns)
            const loss = mse(values, targetReturns);
            // 7. Backward + update
            loss.backward();
            optimizer.step();
            // Report
            if (onEpisode) {
                onEpisode(episode + 1, reward, Array.from(values.contiguousData()));
            }
            // Yield to event loop
            await new Promise((r) => setTimeout(r, 0));
        }
    }

    /**
     * GPU-accelerated SFT training step.
     *
     * Bypasses the synchronous autograd system by manually computing
     * the forward pass (GPU) and LM head gradient (GPU).
     * The backbone is frozen, so we only need grad w.r.t. the LM head weight.
     *
     * Flow:
     *   1. Backbone forward (GPU) → hidden states
     *   2. LM head: logits = hidden @ W^T  (GPU)
     *   3. Cross-entropy loss (CPU — cheap, just softmax + log)
     *   4. Gradient: dL/dW = dL/dlogits^T @ hidden  (GPU)
     *   5. Return loss value + set weight.grad
     */
    /**
     * Run a single GPU-accelerated SFT training step.
     *
     * @param model - PythiaModel with frozen backbone
     * @param lmHeadWeight - The trainable LM head weight tensor [vocabSize, hiddenDim]
     * @param fullIds - Full token sequence (prompt + completion)
     * @param promptLen - Number of prompt tokens (loss computed on completion tokens only)
     * @returns Loss value. Sets lmHeadWeight.grad for the optimizer.
     */
    async function sftStepGPU(model, lmHeadWeight, fullIds, promptLen) {
        const seqLen = fullIds.length;
        const vocabSize = model.config.vocabSize;
        const hiddenDim = model.config.hiddenDim;
        const startPos = promptLen - 1; // First position that predicts a completion token
        const endPos = seqLen - 1; // Last position with a target
        const numTargets = endPos - startPos;
        assert(numTargets > 0, 'No completion tokens to train on');
        // ── 1. Backbone forward → hidden states ─────────────────────────
        // Capture hidden from last layer output (before final LN + LM head)
        let rawHidden = null;
        const hookName = `layers.${model.config.numLayers - 1}.output`;
        const removeHook = model.hooks.add(hookName, (_name, tensor) => {
            rawHidden = tensor;
        });
        // Use GPU forward but we actually need hidden states, not logits.
        // Run forwardGPU which fires hooks.
        if (isWebGPUAvailable()) {
            await model.forwardGPU(fullIds);
        }
        else {
            model.forward(fullIds);
        }
        removeHook();
        if (!rawHidden) {
            throw new Error('[tinyllms] Failed to capture hidden states');
        }
        // Apply final layer norm (CPU — cheap)
        const { layerNorm } = await Promise.resolve().then(function () { return ops; });
        const finalLnW = model._w('final_ln.weight');
        const finalLnB = model._w('final_ln.bias');
        const hidden = layerNorm(rawHidden, finalLnW, finalLnB);
        // hidden: [seqLen, hiddenDim]
        const hiddenData = hidden.contiguousData();
        const wData = lmHeadWeight.contiguousData();
        // ── 2. LM head forward: logits = hidden @ W^T (GPU) ────────────
        // Only compute for the positions we need [startPos..endPos)
        // Extract the relevant hidden states
        const targetHidden = new Float32Array(numTargets * hiddenDim);
        for (let t = 0; t < numTargets; t++) {
            const srcOff = (startPos + t) * hiddenDim;
            const dstOff = t * hiddenDim;
            for (let d = 0; d < hiddenDim; d++) {
                targetHidden[dstOff + d] = hiddenData[srcOff + d];
            }
        }
        // Transpose W for matmul: [vocabSize, hiddenDim] → [hiddenDim, vocabSize]
        const wT = new Float32Array(hiddenDim * vocabSize);
        for (let j = 0; j < vocabSize; j++) {
            for (let k = 0; k < hiddenDim; k++) {
                wT[k * vocabSize + j] = wData[j * hiddenDim + k];
            }
        }
        let logitsData;
        if (shouldUseGPU(numTargets, hiddenDim, vocabSize)) {
            logitsData = await gpuMatmul(targetHidden, wT, numTargets, hiddenDim, vocabSize);
        }
        else {
            // CPU fallback
            logitsData = cpuMatmul(targetHidden, wT, numTargets, hiddenDim, vocabSize);
        }
        // logitsData: [numTargets, vocabSize]
        // ── 3. Cross-entropy loss + gradient w.r.t. logits (CPU) ───────
        const targets = fullIds.slice(startPos + 1, endPos + 1);
        const gradLogits = new Float32Array(numTargets * vocabSize);
        let totalLoss = 0;
        for (let t = 0; t < numTargets; t++) {
            const offset = t * vocabSize;
            const targetIdx = targets[t];
            // Numerically stable softmax
            let maxVal = -Infinity;
            for (let j = 0; j < vocabSize; j++) {
                if (logitsData[offset + j] > maxVal)
                    maxVal = logitsData[offset + j];
            }
            let sumExp = 0;
            for (let j = 0; j < vocabSize; j++) {
                const e = Math.exp(logitsData[offset + j] - maxVal);
                gradLogits[offset + j] = e; // temporarily store softmax probs
                sumExp += e;
            }
            for (let j = 0; j < vocabSize; j++) {
                gradLogits[offset + j] /= sumExp;
            }
            // Loss: -log(prob[target])
            totalLoss -= Math.log(gradLogits[offset + targetIdx] + 1e-10);
            // Gradient: softmax - one_hot(target), scaled by 1/numTargets
            for (let j = 0; j < vocabSize; j++) {
                gradLogits[offset + j] = (gradLogits[offset + j] - (j === targetIdx ? 1 : 0)) / numTargets;
            }
        }
        const loss = totalLoss / numTargets;
        // ── 4. LM head weight gradient: dL/dW = gradLogits^T @ hidden (GPU)
        // gradLogits: [numTargets, vocabSize], targetHidden: [numTargets, hiddenDim]
        // dL/dW[j, k] = sum_t gradLogits[t, j] * hidden[t, k]
        // = gradLogits^T @ hidden → [vocabSize, hiddenDim]
        let gradWData;
        if (shouldUseGPU(vocabSize, numTargets, hiddenDim)) {
            // gradLogits^T: [vocabSize, numTargets]
            const gradLogitsT = new Float32Array(vocabSize * numTargets);
            for (let t = 0; t < numTargets; t++) {
                for (let j = 0; j < vocabSize; j++) {
                    gradLogitsT[j * numTargets + t] = gradLogits[t * vocabSize + j];
                }
            }
            gradWData = await gpuMatmul(gradLogitsT, targetHidden, vocabSize, numTargets, hiddenDim);
        }
        else {
            // CPU: j-t-k order for cache-friendliness
            gradWData = new Float32Array(vocabSize * hiddenDim);
            for (let j = 0; j < vocabSize; j++) {
                for (let t = 0; t < numTargets; t++) {
                    const gVal = gradLogits[t * vocabSize + j];
                    const hOff = t * hiddenDim;
                    const wOff = j * hiddenDim;
                    for (let k = 0; k < hiddenDim; k++) {
                        gradWData[wOff + k] += gVal * targetHidden[hOff + k];
                    }
                }
            }
        }
        // Set gradient on the weight tensor
        lmHeadWeight.grad = new Tensor(gradWData, [...lmHeadWeight.shape]);
        return { loss };
    }
    /** Simple CPU matmul for fallback. */
    function cpuMatmul(a, b, M, K, N) {
        const c = new Float32Array(M * N);
        for (let i = 0; i < M; i++) {
            for (let k = 0; k < K; k++) {
                const aVal = a[i * K + k];
                const bOff = k * N;
                const cOff = i * N;
                for (let j = 0; j < N; j++) {
                    c[cOff + j] += aVal * b[bOff + j];
                }
            }
        }
        return c;
    }

    /**
     * Shared canvas utilities for viz components.
     */
    /**
     * Create and configure a HiDPI-aware canvas inside a container.
     * Returns the canvas, context, device pixel ratio, and logical dimensions.
     */
    function createCanvas(container, width, height) {
        const canvas = document.createElement('canvas');
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        container.appendChild(canvas);
        return { canvas, ctx, dpr, width, height };
    }
    /** Clear the entire canvas. */
    function clearCanvas(ctx, width, height) {
        ctx.clearRect(0, 0, width, height);
    }

    /**
     * Color utilities for visualization components.
     * All colors are [r, g, b] tuples in 0-255 range.
     */
    /** Default color palette. */
    const COLORS = {
        white: [255, 255, 255],
        accentBlue: [37, 99, 235], // #2563eb
        negativeRed: [220, 38, 38], // #dc2626
        positiveGreen: [22, 163, 74], // #16a34a
        gray: [136, 136, 136], // #888
        text: [26, 26, 26], // #1a1a1a
        muted: [85, 85, 85], // #555
        border: [224, 224, 224], // #e0e0e0
        cardBg: [245, 245, 240], // #f5f5f0
    };
    /** Linear interpolation between two colors. t in [0, 1]. */
    function lerpColor(a, b, t) {
        const tc = Math.max(0, Math.min(1, t));
        return [
            Math.round(a[0] + (b[0] - a[0]) * tc),
            Math.round(a[1] + (b[1] - a[1]) * tc),
            Math.round(a[2] + (b[2] - a[2]) * tc),
        ];
    }
    /** Convert RGB tuple to CSS string. */
    function rgbString(color, alpha = 1) {
        if (alpha < 1) {
            return `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
        }
        return `rgb(${color[0]},${color[1]},${color[2]})`;
    }
    /**
     * Diverging color scale: negative (red) → zero (white) → positive (green).
     * value should be in [-1, 1] (caller normalizes).
     */
    function divergingColor(value, negColor = COLORS.negativeRed, zeroColor = COLORS.white, posColor = COLORS.positiveGreen) {
        if (value <= 0) {
            return lerpColor(negColor, zeroColor, value + 1); // -1→neg, 0→zero
        }
        return lerpColor(zeroColor, posColor, value); // 0→zero, 1→pos
    }
    /**
     * Sequential color scale: low (white) → high (accent).
     * value in [0, 1].
     */
    function sequentialColor(value, low = COLORS.white, high = COLORS.accentBlue) {
        return lerpColor(low, high, value);
    }

    /**
     * Renders attention weights as a heatmap with head selection tabs.
     */
    class AttentionViz {
        _container;
        _canvas = null;
        _ctx = null;
        _width;
        _height;
        _highColor;
        _data = null;
        _selectedHead = 0;
        _tabsEl = null;
        constructor(config) {
            this._container = config.container;
            this._width = config.width ?? 400;
            this._height = config.height ?? 400;
            this._highColor = config.highColor ?? COLORS.accentBlue;
        }
        /** Update the visualization with new data. */
        update(data) {
            this._data = data;
            if (!this._canvas)
                this._setup(data.shape[0]);
            this._render();
        }
        /** Select which attention head to display. */
        selectHead(head) {
            this._selectedHead = head;
            this._updateTabs();
            this._render();
        }
        destroy() {
            if (this._canvas)
                this._canvas.remove();
            if (this._tabsEl)
                this._tabsEl.remove();
            this._canvas = null;
            this._ctx = null;
            this._tabsEl = null;
        }
        _setup(numHeads) {
            // Create head selection tabs
            this._tabsEl = document.createElement('div');
            this._tabsEl.style.cssText = 'display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap;';
            for (let h = 0; h < numHeads; h++) {
                const btn = document.createElement('button');
                btn.textContent = `Head ${h}`;
                btn.style.cssText = 'padding:2px 8px;border:1px solid #ccc;border-radius:4px;cursor:pointer;font-size:12px;background:#fff;';
                btn.addEventListener('click', () => this.selectHead(h));
                this._tabsEl.appendChild(btn);
            }
            this._container.appendChild(this._tabsEl);
            this._updateTabs();
            // Create canvas
            const setup = createCanvas(this._container, this._width, this._height);
            this._canvas = setup.canvas;
            this._ctx = setup.ctx;
        }
        _updateTabs() {
            if (!this._tabsEl)
                return;
            const buttons = this._tabsEl.querySelectorAll('button');
            buttons.forEach((btn, i) => {
                btn.style.background = i === this._selectedHead ? rgbString(this._highColor) : '#fff';
                btn.style.color = i === this._selectedHead ? '#fff' : '#333';
            });
        }
        _render() {
            if (!this._ctx || !this._data)
                return;
            const { tokens, weights, shape } = this._data;
            const [numHeads, seqLen] = shape;
            const ctx = this._ctx;
            const head = this._selectedHead;
            clearCanvas(ctx, this._width, this._height);
            // Layout: labels on left/top, heatmap in remaining space
            const labelWidth = 60;
            const labelHeight = 60;
            const heatW = this._width - labelWidth;
            const heatH = this._height - labelHeight;
            const cellW = heatW / seqLen;
            const cellH = heatH / seqLen;
            // Draw heatmap cells
            const headOffset = head * seqLen * seqLen;
            for (let i = 0; i < seqLen; i++) {
                for (let j = 0; j < seqLen; j++) {
                    const w = weights[headOffset + i * seqLen + j];
                    const color = sequentialColor(w, COLORS.white, this._highColor);
                    ctx.fillStyle = rgbString(color);
                    ctx.fillRect(labelWidth + j * cellW, labelHeight + i * cellH, cellW, cellH);
                }
            }
            // Draw grid lines
            ctx.strokeStyle = rgbString(COLORS.border);
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= seqLen; i++) {
                ctx.beginPath();
                ctx.moveTo(labelWidth + i * cellW, labelHeight);
                ctx.lineTo(labelWidth + i * cellW, this._height);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(labelWidth, labelHeight + i * cellH);
                ctx.lineTo(this._width, labelHeight + i * cellH);
                ctx.stroke();
            }
            // Draw labels
            ctx.fillStyle = rgbString(COLORS.text);
            ctx.font = '10px monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = 0; i < seqLen; i++) {
                const label = tokens[i] ?? String(i);
                const truncated = label.length > 6 ? label.slice(0, 6) : label;
                // Row labels (left)
                ctx.save();
                ctx.textAlign = 'right';
                ctx.fillText(truncated, labelWidth - 4, labelHeight + i * cellH + cellH / 2);
                ctx.restore();
                // Column labels (top, rotated)
                ctx.save();
                ctx.translate(labelWidth + i * cellW + cellW / 2, labelHeight - 4);
                ctx.rotate(-Math.PI / 4);
                ctx.textAlign = 'right';
                ctx.fillText(truncated, 0, 0);
                ctx.restore();
            }
        }
    }

    /**
     * Top-k token probability horizontal bar chart.
     */
    class LogitsViz {
        _container;
        _canvas = null;
        _ctx = null;
        _width;
        _height;
        _barColor;
        constructor(config) {
            this._container = config.container;
            this._width = config.width ?? 400;
            this._height = config.height ?? 300;
            this._barColor = config.barColor ?? COLORS.accentBlue;
        }
        update(data) {
            if (!this._canvas) {
                const setup = createCanvas(this._container, this._width, this._height);
                this._canvas = setup.canvas;
                this._ctx = setup.ctx;
            }
            this._render(data);
        }
        destroy() {
            if (this._canvas)
                this._canvas.remove();
            this._canvas = null;
            this._ctx = null;
        }
        _render(data) {
            const ctx = this._ctx;
            const { tokens, probs, topK } = data;
            clearCanvas(ctx, this._width, this._height);
            // Sort by probability, take top-k
            const indices = Array.from({ length: probs.length }, (_, i) => i);
            indices.sort((a, b) => probs[b] - probs[a]);
            const topIndices = indices.slice(0, topK);
            const labelWidth = 80;
            const probWidth = 50;
            const barAreaWidth = this._width - labelWidth - probWidth - 20;
            const barHeight = Math.min(24, (this._height - 20) / topK);
            const barGap = 4;
            const maxProb = topIndices.length > 0 ? probs[topIndices[0]] : 1;
            for (let i = 0; i < topIndices.length; i++) {
                const idx = topIndices[i];
                const prob = probs[idx];
                const token = tokens[idx] ?? `[${idx}]`;
                const y = 10 + i * (barHeight + barGap);
                // Token label
                ctx.fillStyle = rgbString(COLORS.text);
                ctx.font = '12px monospace';
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                const truncated = token.length > 10 ? token.slice(0, 10) : token;
                ctx.fillText(truncated, labelWidth - 4, y + barHeight / 2);
                // Bar
                const barW = (prob / maxProb) * barAreaWidth;
                ctx.fillStyle = rgbString(this._barColor, 0.8);
                ctx.fillRect(labelWidth, y, barW, barHeight);
                // Probability text
                ctx.fillStyle = rgbString(COLORS.muted);
                ctx.font = '11px monospace';
                ctx.textAlign = 'left';
                ctx.fillText((prob * 100).toFixed(1) + '%', labelWidth + barW + 4, y + barHeight / 2);
            }
        }
    }

    /**
     * Live-updating multi-series line chart for loss curves.
     * Supports multiple named series with auto-scaling Y axis.
     */
    class LossCurveViz {
        _container;
        _canvas = null;
        _ctx = null;
        _width;
        _height;
        _series = new Map();
        _nextColorIdx = 0;
        static PALETTE = [
            COLORS.accentBlue,
            COLORS.negativeRed,
            COLORS.positiveGreen,
            COLORS.gray,
            [168, 85, 247], // purple
            [234, 88, 12], // orange
        ];
        constructor(config) {
            this._container = config.container;
            this._width = config.width ?? 500;
            this._height = config.height ?? 300;
        }
        /**
         * Add a data point to a named series.
         * Creates the series if it does not exist.
         */
        addPoint(seriesName, step, value) {
            if (!this._canvas) {
                const setup = createCanvas(this._container, this._width, this._height);
                this._canvas = setup.canvas;
                this._ctx = setup.ctx;
            }
            let series = this._series.get(seriesName);
            if (!series) {
                const color = LossCurveViz.PALETTE[this._nextColorIdx % LossCurveViz.PALETTE.length];
                this._nextColorIdx++;
                series = { name: seriesName, color, points: [] };
                this._series.set(seriesName, series);
            }
            series.points.push({ step, value });
            this._render();
        }
        /** Clear all series data. */
        clear() {
            this._series.clear();
            this._nextColorIdx = 0;
            if (this._ctx)
                clearCanvas(this._ctx, this._width, this._height);
        }
        destroy() {
            if (this._canvas)
                this._canvas.remove();
            this._canvas = null;
            this._ctx = null;
        }
        _render() {
            const ctx = this._ctx;
            clearCanvas(ctx, this._width, this._height);
            if (this._series.size === 0)
                return;
            // Compute global bounds
            let minStep = Infinity, maxStep = -Infinity;
            let minVal = Infinity, maxVal = -Infinity;
            for (const series of this._series.values()) {
                for (const p of series.points) {
                    if (p.step < minStep)
                        minStep = p.step;
                    if (p.step > maxStep)
                        maxStep = p.step;
                    if (p.value < minVal)
                        minVal = p.value;
                    if (p.value > maxVal)
                        maxVal = p.value;
                }
            }
            // Add padding to Y range
            const yRange = maxVal - minVal || 1;
            minVal -= yRange * 0.05;
            maxVal += yRange * 0.05;
            const xRange = maxStep - minStep || 1;
            // Layout
            const margin = { top: 20, right: 20, bottom: 30, left: 50 };
            const plotW = this._width - margin.left - margin.right;
            const plotH = this._height - margin.top - margin.bottom;
            const toX = (step) => margin.left + ((step - minStep) / xRange) * plotW;
            const toY = (val) => margin.top + (1 - (val - minVal) / (maxVal - minVal)) * plotH;
            // Draw axes
            ctx.strokeStyle = rgbString(COLORS.border);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin.left, margin.top);
            ctx.lineTo(margin.left, margin.top + plotH);
            ctx.lineTo(margin.left + plotW, margin.top + plotH);
            ctx.stroke();
            // Y-axis ticks
            ctx.fillStyle = rgbString(COLORS.muted);
            ctx.font = '10px monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            const numYTicks = 5;
            for (let i = 0; i <= numYTicks; i++) {
                const val = minVal + (i / numYTicks) * (maxVal - minVal);
                const y = toY(val);
                ctx.fillText(val.toPrecision(3), margin.left - 4, y);
                // Grid line
                ctx.strokeStyle = rgbString(COLORS.border, 0.3);
                ctx.beginPath();
                ctx.moveTo(margin.left, y);
                ctx.lineTo(margin.left + plotW, y);
                ctx.stroke();
            }
            // X-axis label
            ctx.fillStyle = rgbString(COLORS.muted);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Step', margin.left + plotW / 2, this._height - 10);
            // Draw each series
            for (const series of this._series.values()) {
                if (series.points.length < 2)
                    continue;
                ctx.strokeStyle = rgbString(series.color);
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                const sorted = [...series.points].sort((a, b) => a.step - b.step);
                ctx.moveTo(toX(sorted[0].step), toY(sorted[0].value));
                for (let i = 1; i < sorted.length; i++) {
                    ctx.lineTo(toX(sorted[i].step), toY(sorted[i].value));
                }
                ctx.stroke();
            }
            // Legend
            if (this._series.size > 1) {
                let legendY = margin.top + 4;
                ctx.font = '10px monospace';
                for (const series of this._series.values()) {
                    ctx.fillStyle = rgbString(series.color);
                    ctx.fillRect(margin.left + 8, legendY, 12, 3);
                    ctx.fillStyle = rgbString(COLORS.text);
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'top';
                    ctx.fillText(series.name, margin.left + 24, legendY - 3);
                    legendY += 14;
                }
            }
        }
    }

    /**
     * Renders tokens as inline <span> elements with colored backgrounds.
     * Uses sequential color scale for probabilities/attention (0→1),
     * diverging scale for values (negative→zero→positive).
     */
    class TokenViz {
        _container;
        _wrapper = null;
        constructor(config) {
            this._container = config.container;
        }
        update(data) {
            this.destroy();
            this._wrapper = document.createElement('div');
            this._wrapper.style.cssText = 'line-height:1.8;font-family:monospace;font-size:14px;';
            const { tokens, values, mode } = data;
            // Normalize values for color mapping
            let normalizedValues;
            if (mode === 'value') {
                // Diverging: normalize to [-1, 1] based on max absolute value
                const maxAbs = values.reduce((mx, v) => Math.max(mx, Math.abs(v)), 0) || 1;
                normalizedValues = values.map((v) => v / maxAbs);
            }
            else {
                // Sequential: normalize to [0, 1]
                const minV = Math.min(...values);
                const maxV = Math.max(...values);
                const range = maxV - minV || 1;
                normalizedValues = values.map((v) => (v - minV) / range);
            }
            for (let i = 0; i < tokens.length; i++) {
                const span = document.createElement('span');
                span.textContent = tokens[i] ?? '';
                const nv = normalizedValues[i] ?? 0;
                let bgColor;
                if (mode === 'value') {
                    bgColor = divergingColor(nv);
                }
                else if (mode === 'probability') {
                    bgColor = sequentialColor(nv, COLORS.white, COLORS.accentBlue);
                }
                else {
                    // attention
                    bgColor = sequentialColor(nv, COLORS.white, COLORS.accentBlue);
                }
                // Choose text color based on background brightness
                const brightness = 0.299 * bgColor[0] + 0.587 * bgColor[1] + 0.114 * bgColor[2];
                const textColor = brightness < 128 ? '#fff' : '#1a1a1a';
                span.style.cssText = `
        display:inline-block;
        padding:2px 4px;
        margin:1px;
        border-radius:3px;
        background:${rgbString(bgColor)};
        color:${textColor};
      `;
                // Tooltip with value
                span.title = `${tokens[i]}: ${values[i]?.toFixed(4)}`;
                this._wrapper.appendChild(span);
            }
            this._container.appendChild(this._wrapper);
        }
        destroy() {
            if (this._wrapper) {
                this._wrapper.remove();
                this._wrapper = null;
            }
        }
    }

    /**
     * Per-token value estimates as colored bars above/below a token strip.
     * Green (positive) above center, red (negative) below.
     * Supports multiple snapshots with a step slider.
     */
    class ValueMapViz {
        _container;
        _canvas = null;
        _ctx = null;
        _width;
        _height;
        _tokens = [];
        _snapshots = [];
        _selectedIdx = 0;
        _sliderEl = null;
        _labelEl = null;
        constructor(config) {
            this._container = config.container;
            this._width = config.width ?? 500;
            this._height = config.height ?? 200;
        }
        /** Set the token strings (fixed across snapshots). */
        setTokens(tokens) {
            this._tokens = tokens;
        }
        /** Add a snapshot of value estimates at a given training step. */
        addSnapshot(snapshot) {
            this._snapshots.push(snapshot);
            if (!this._canvas)
                this._setup();
            // Update slider range
            if (this._sliderEl) {
                this._sliderEl.max = String(this._snapshots.length - 1);
                this._sliderEl.value = String(this._snapshots.length - 1);
                this._selectedIdx = this._snapshots.length - 1;
            }
            this._render();
        }
        /** Clear all snapshots. */
        clear() {
            this._snapshots = [];
            this._selectedIdx = 0;
            if (this._sliderEl) {
                this._sliderEl.max = '0';
                this._sliderEl.value = '0';
            }
            if (this._ctx)
                clearCanvas(this._ctx, this._width, this._height);
        }
        destroy() {
            if (this._canvas)
                this._canvas.remove();
            if (this._sliderEl)
                this._sliderEl.parentElement?.remove();
            if (this._labelEl)
                this._labelEl.remove();
            this._canvas = null;
            this._ctx = null;
            this._sliderEl = null;
            this._labelEl = null;
        }
        _setup() {
            // Canvas
            const setup = createCanvas(this._container, this._width, this._height);
            this._canvas = setup.canvas;
            this._ctx = setup.ctx;
            // Slider row
            const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:8px;margin-top:4px;';
            this._labelEl = document.createElement('span');
            this._labelEl.style.cssText = 'font:11px monospace;color:#555;min-width:60px;';
            this._sliderEl = document.createElement('input');
            this._sliderEl.type = 'range';
            this._sliderEl.min = '0';
            this._sliderEl.max = '0';
            this._sliderEl.value = '0';
            this._sliderEl.style.cssText = 'flex:1;';
            this._sliderEl.addEventListener('input', () => {
                this._selectedIdx = parseInt(this._sliderEl.value, 10);
                this._render();
            });
            row.appendChild(this._labelEl);
            row.appendChild(this._sliderEl);
            this._container.appendChild(row);
        }
        _render() {
            const ctx = this._ctx;
            clearCanvas(ctx, this._width, this._height);
            if (this._snapshots.length === 0 || this._tokens.length === 0)
                return;
            const snapshot = this._snapshots[this._selectedIdx];
            if (!snapshot)
                return;
            // Update label
            if (this._labelEl) {
                this._labelEl.textContent = `Step ${snapshot.step}`;
            }
            const numTokens = this._tokens.length;
            const margin = { top: 10, bottom: 30, left: 10, right: 10 };
            const plotW = this._width - margin.left - margin.right;
            const plotH = this._height - margin.top - margin.bottom;
            const barWidth = plotW / numTokens;
            const centerY = margin.top + plotH / 2;
            // Find max absolute value for normalization
            const maxAbs = snapshot.values.reduce((mx, v) => Math.max(mx, Math.abs(v)), 0) || 1;
            // Draw center line
            ctx.strokeStyle = rgbString(COLORS.border);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(margin.left, centerY);
            ctx.lineTo(margin.left + plotW, centerY);
            ctx.stroke();
            // Draw bars
            for (let i = 0; i < numTokens; i++) {
                const v = snapshot.values[i] ?? 0;
                const normalized = v / maxAbs; // in [-1, 1]
                const barH = Math.abs(normalized) * (plotH / 2);
                const x = margin.left + i * barWidth;
                const color = divergingColor(normalized);
                ctx.fillStyle = rgbString(color);
                if (v >= 0) {
                    ctx.fillRect(x + 1, centerY - barH, barWidth - 2, barH);
                }
                else {
                    ctx.fillRect(x + 1, centerY, barWidth - 2, barH);
                }
            }
            // Draw token labels at bottom
            ctx.fillStyle = rgbString(COLORS.text);
            ctx.font = '9px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            for (let i = 0; i < numTokens; i++) {
                const label = this._tokens[i] ?? '';
                const truncated = label.length > 4 ? label.slice(0, 4) : label;
                const x = margin.left + i * barWidth + barWidth / 2;
                ctx.fillText(truncated, x, this._height - margin.bottom + 4);
            }
        }
    }

    exports.Adam = Adam;
    exports.AttentionViz = AttentionViz;
    exports.COLORS = COLORS;
    exports.ClassifierHead = ClassifierHead;
    exports.Dataset = Dataset;
    exports.HookRegistry = HookRegistry;
    exports.LogitsViz = LogitsViz;
    exports.LossCurveViz = LossCurveViz;
    exports.PythiaModel = PythiaModel;
    exports.RewardHead = RewardHead;
    exports.SGD = SGD;
    exports.Tensor = Tensor;
    exports.TokenViz = TokenViz;
    exports.Tokenizer = Tokenizer;
    exports.ValueHead = ValueHead;
    exports.ValueMapViz = ValueMapViz;
    exports.add = add;
    exports.assert = assert;
    exports.backward = backward;
    exports.bradleyTerry = bradleyTerry;
    exports.broadcastShape = broadcastShape;
    exports.cat = cat;
    exports.causalMask = causalMask;
    exports.computeStrides = computeStrides;
    exports.crossEntropy = crossEntropy;
    exports.div = div;
    exports.divergingColor = divergingColor;
    exports.embedding = embedding;
    exports.exp = exp;
    exports.gelu = gelu;
    exports.getBackend = getBackend;
    exports.initWebGPU = initWebGPU;
    exports.isWebGPUAvailable = isWebGPUAvailable;
    exports.layerNorm = layerNorm;
    exports.lerpColor = lerpColor;
    exports.linear = linear;
    exports.loadTokenizer = loadTokenizer;
    exports.loadWeights = loadWeights;
    exports.matmul = matmul;
    exports.max = max;
    exports.mean = mean;
    exports.mse = mse;
    exports.mul = mul;
    exports.neg = neg;
    exports.reductionAxes = reductionAxes;
    exports.reshape = reshape;
    exports.rgbString = rgbString;
    exports.rope = rope;
    exports.sequentialColor = sequentialColor;
    exports.sftStepGPU = sftStepGPU;
    exports.shapeEqual = shapeEqual;
    exports.shapeSize = shapeSize;
    exports.slice = slice;
    exports.softmax = softmax;
    exports.sub = sub;
    exports.sum = sum;
    exports.train = train;
    exports.trainValueFunction = trainValueFunction;
    exports.transpose = transpose;

}));
//# sourceMappingURL=tinyllms.umd.js.map
