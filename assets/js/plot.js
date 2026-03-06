/*
 * plot.js — Lightweight canvas plotting for blog posts.
 *
 * Usage:
 *   <canvas id="my-plot" style="width:100%;height:auto;border-radius:4px;display:block;"
 *           data-plot="silu"></canvas>
 *   <script src="/assets/js/plot.js"></script>
 *
 * Built-in presets: "silu", "sigmoid", "relu", "gelu", "tanh"
 * Or define custom plots via BlogPlot.draw(id, fns, opts).
 *
 * API:
 *   BlogPlot.draw(canvasId, functions, options)
 *     functions: [{ f: x => ..., color, width, dash, label, labelX, labelBold }]
 *     options:   { xMin, xMax, yMin, yMax, markPoints: [{x, label}], bg }
 *
 *   BlogPlot.preset(canvasId, presetName)
 *     Draws a named preset. Called automatically for canvases with data-plot attribute.
 */

(function () {
    'use strict';

    // Common activation functions
    var sigmoid = function (x) { return 1 / (1 + Math.exp(-x)); };
    var silu = function (x) { return x * sigmoid(x); };
    var relu = function (x) { return Math.max(0, x); };
    var gelu = function (x) {
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
    };
    var tanhFn = function (x) { return Math.tanh(x); };

    // Presets
    var PRESETS = {
        silu: {
            fns: [
                { f: relu, color: '#ccc', width: 1.5, dash: [4, 3], label: 'ReLU', labelX: 3 },
                { f: silu, color: '#2563eb', width: 2.5, label: 'SiLU', labelX: 2.5, labelBold: true }
            ],
            opts: { xMin: -5, xMax: 5, yMin: -1, yMax: 5, markPoints: [{ x: -1.28, label: 'dip \u2248 -0.28' }] }
        },
        sigmoid: {
            fns: [
                { f: sigmoid, color: '#2563eb', width: 2.5, label: '\u03C3(x)', labelX: 2, labelBold: true }
            ],
            opts: { xMin: -6, xMax: 6, yMin: -0.2, yMax: 1.2 }
        },
        relu: {
            fns: [
                { f: relu, color: '#2563eb', width: 2.5, label: 'ReLU', labelX: 3, labelBold: true }
            ],
            opts: { xMin: -4, xMax: 4, yMin: -1, yMax: 4 }
        },
        gelu: {
            fns: [
                { f: relu, color: '#ccc', width: 1.5, dash: [4, 3], label: 'ReLU', labelX: 3 },
                { f: gelu, color: '#2563eb', width: 2.5, label: 'GELU', labelX: 2.5, labelBold: true }
            ],
            opts: { xMin: -5, xMax: 5, yMin: -1, yMax: 5, markPoints: [{ x: -0.75, label: 'dip' }] }
        },
        tanh: {
            fns: [
                { f: tanhFn, color: '#2563eb', width: 2.5, label: 'tanh', labelX: 2, labelBold: true }
            ],
            opts: { xMin: -4, xMax: 4, yMin: -1.5, yMax: 1.5 }
        }
    };

    function draw(id, fns, opts) {
        var c = document.getElementById(id);
        if (!c) return;

        var dpr = window.devicePixelRatio || 1;
        var w = 290, h = 150;
        c.width = w * dpr;
        c.height = h * dpr;
        c.style.height = 'auto';
        c.style.aspectRatio = w + '/' + h;

        var ctx = c.getContext('2d');
        ctx.scale(dpr, dpr);

        var xMin = opts.xMin, xMax = opts.xMax;
        var yMin = opts.yMin, yMax = opts.yMax;
        var pad = { l: 28, r: 10, t: 10, b: 22 };
        var pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

        function toX(x) { return pad.l + (x - xMin) / (xMax - xMin) * pw; }
        function toY(y) { return pad.t + (1 - (y - yMin) / (yMax - yMin)) * ph; }

        // Background
        ctx.fillStyle = opts.bg || '#fafafa';
        ctx.fillRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 0.5;
        for (var gy = Math.ceil(yMin); gy <= yMax; gy++) {
            ctx.beginPath();
            ctx.moveTo(toX(xMin), toY(gy));
            ctx.lineTo(toX(xMax), toY(gy));
            ctx.stroke();
        }

        // Axes
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(toX(xMin), toY(0)); ctx.lineTo(toX(xMax), toY(0)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(toX(0), toY(yMin)); ctx.lineTo(toX(0), toY(yMax)); ctx.stroke();

        // Axis tick labels
        ctx.fillStyle = '#bbb';
        ctx.font = '9px Georgia';
        ctx.textAlign = 'center';
        ctx.fillText(xMin, toX(xMin), toY(0) + 13);
        ctx.fillText(xMax, toX(xMax), toY(0) + 13);
        ctx.fillText('0', toX(0) - 9, toY(0) + 13);
        ctx.textAlign = 'right';
        for (var ty = Math.ceil(yMin); ty <= yMax; ty++) {
            if (ty !== 0) ctx.fillText(ty, toX(xMin) - 4, toY(ty) + 3);
        }

        // Plot each function
        var steps = 200;
        fns.forEach(function (fn) {
            ctx.beginPath();
            ctx.strokeStyle = fn.color || '#2563eb';
            ctx.lineWidth = fn.width || 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            if (fn.dash) ctx.setLineDash(fn.dash); else ctx.setLineDash([]);

            for (var i = 0; i <= steps; i++) {
                var x = xMin + (xMax - xMin) * i / steps;
                var y = Math.max(yMin, Math.min(yMax, fn.f(x)));
                i === 0 ? ctx.moveTo(toX(x), toY(y)) : ctx.lineTo(toX(x), toY(y));
            }
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            if (fn.label) {
                var lx = fn.labelX !== undefined ? fn.labelX : xMax * 0.6;
                ctx.fillStyle = fn.color || '#2563eb';
                ctx.font = (fn.labelBold ? 'bold ' : '') + '10px Georgia';
                ctx.textAlign = 'left';
                ctx.fillText(fn.label, toX(lx), toY(fn.f(lx)) - 6);
            }
        });

        // Mark specific points (e.g., dips, inflection points)
        if (opts.markPoints) {
            opts.markPoints.forEach(function (pt) {
                // Find the function value: use the last (primary) function
                var primaryFn = fns[fns.length - 1].f;
                var py = primaryFn(pt.x);
                ctx.beginPath();
                ctx.arc(toX(pt.x), toY(py), 3, 0, Math.PI * 2);
                ctx.fillStyle = fns[fns.length - 1].color || '#2563eb';
                ctx.fill();
                if (pt.label) {
                    ctx.fillStyle = '#999';
                    ctx.font = '8px Georgia';
                    ctx.textAlign = 'right';
                    ctx.fillText(pt.label, toX(pt.x) - 6, toY(py) + 3);
                }
            });
        }
    }

    function preset(id, name) {
        var p = PRESETS[name];
        if (p) draw(id, p.fns, p.opts);
    }

    // Auto-init: find all canvases with data-plot and draw them
    function autoInit() {
        var canvases = document.querySelectorAll('canvas[data-plot]');
        canvases.forEach(function (c) {
            var name = c.getAttribute('data-plot');
            if (c.id && PRESETS[name]) {
                preset(c.id, name);
                // Re-draw on first hover of parent .eq-term (tooltip may be hidden at load)
                var term = c.closest('.eq-term');
                if (term) {
                    term.addEventListener('mouseenter', function () {
                        setTimeout(function () { preset(c.id, name); }, 10);
                    }, { once: true });
                }
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', autoInit);
    } else {
        autoInit();
    }

    // Export for manual use
    window.BlogPlot = { draw: draw, preset: preset, fns: { sigmoid: sigmoid, silu: silu, relu: relu, gelu: gelu, tanh: tanhFn } };
})();
