/*
 * spec-decoding-anim.js - Step-by-step speculative decoding animation.
 *
 * Renders two decoding schedules side by side on a shared clock: plain
 * autoregressive on the left, speculative with k drafted tokens on the right.
 * Both produce the identical token sequence; the right finishes sooner.
 *
 * Usage - drop this markup on the page, then load the script:
 *
 *   <div class="diagram" style="padding: 14px 14px 10px; overflow: visible;">
 *       <canvas id="spec-canvas" width="960" height="600"
 *               style="width:100%; height:auto;"></canvas>
 *       <div class="anim-controls">
 *           <button id="spec-prev"  class="anim-btn secondary">Prev</button>
 *           <button id="spec-next"  class="anim-btn">Next step</button>
 *           <button id="spec-play"  class="anim-btn secondary">Play</button>
 *           <button id="spec-reset" class="anim-btn secondary">Reset</button>
 *           <span id="spec-count" class="anim-count"></span>
 *       </div>
 *       <p id="spec-caption" class="anim-caption"></p>
 *   </div>
 *   <script src="/assets/js/spec-decoding-anim.js"></script>
 *
 * Styles for .anim-btn / .anim-controls / .anim-caption live in
 * /assets/css/blog-components.css. Mounts on DOMContentLoaded, or immediately
 * if the DOM is already parsed, and no-ops when #spec-canvas is absent.
 */

(function () {
    'use strict';

    function mount() {
        var canvas = document.getElementById('spec-canvas');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var nextBtn = document.getElementById('spec-next');
        var prevBtn = document.getElementById('spec-prev');
        var playBtn = document.getElementById('spec-play');
        var resetBtn = document.getElementById('spec-reset');
        var caption = document.getElementById('spec-caption');
        var countEl = document.getElementById('spec-count');
        if (!nextBtn || !prevBtn || !playBtn || !resetBtn || !caption || !countEl) return;

        // ---- Cost model, in abstract time units ----
        var T_TARGET = 40;   // one large-model forward pass
        var T_DRAFT = 5;     // one drafter step (c = 0.125)
        var K = 4;           // drafted tokens per round

        // ---- The output both sides produce ----
        var OUT = ["the", "GPU", "is", "memory", "-bound", ",", "not", "compute",
                   "-bound", ".", "Verifying", "many", "tokens", "costs", "the",
                   "same", "as", "one", "."];

        // ---- Scripted rounds for the speculative side ----
        // guesses: what the drafter proposes; accept: how many survive verification
        var ROUNDS = [
            { guesses: ["the", "GPU", "is", "model"], accept: 3 },
            { guesses: ["-bound", ",", "not", "compute"], accept: 4 },
            { guesses: [".", "Verifying", "the", "batch"], accept: 2 },
            { guesses: ["tokens", "costs", "the", "same"], accept: 4 },
            { guesses: ["one", "extra", "token", "here"], accept: 1 }
        ];

        // ---- Build both schedules up front, so rendering is a pure function of time ----
        var vanilla = { segs: [], total: 0 };
        for (var i = 0; i < OUT.length; i++) {
            vanilla.segs.push({ t0: i * T_TARGET, t1: (i + 1) * T_TARGET, kind: 'target', emits: i });
        }
        vanilla.total = OUT.length * T_TARGET;

        var spec = { segs: [], rounds: [], total: 0 };
        (function buildSpec() {
            var cursor = 0, committed = 0;
            for (var r = 0; r < ROUNDS.length; r++) {
                var R = ROUNDS[r];
                var draftSegs = [];
                for (var j = 0; j < K; j++) {
                    var s = { t0: cursor, t1: cursor + T_DRAFT, kind: 'draft', round: r, guess: j };
                    cursor += T_DRAFT;
                    spec.segs.push(s);
                    draftSegs.push(s);
                }
                var v = { t0: cursor, t1: cursor + T_TARGET, kind: 'target', round: r };
                cursor += T_TARGET;
                spec.segs.push(v);

                // committed tokens: `accept` from the draft, then one from the target
                var fromDraft = [];
                for (var a = 0; a < R.accept; a++) fromDraft.push(committed + a);
                var fromTarget = committed + R.accept;
                committed += R.accept + 1;

                spec.rounds.push({
                    guesses: R.guesses,
                    accept: R.accept,
                    bonus: R.accept === K,
                    draftSegs: draftSegs,
                    verify: v,
                    fromDraft: fromDraft,
                    fromTarget: fromTarget
                });
            }
            spec.total = cursor;
        })();

        // ---- Layout ----
        var W = canvas.width, H = canvas.height;
        var PW = 452, LX = 12, RX = 496;
        var TL_Y = 152, TL_H = 26, TL_W = 424, TL_X_OFF = 12;
        var PPT = TL_W / vanilla.total;      // pixels per time unit, shared by both panels
        var PASS_Y = 270, PASS_H = 26;
        var OUTQ_Y = 330, BOX_H = 24, LINE_H = 30;

        var GREEN_F = '#f0fdf4', GREEN_S = '#16a34a', GREEN_T = '#166534';
        var RED_F = '#fff5f5', RED_S = '#dc2626', RED_T = '#991b1b';
        var BLUE_F = '#f0f4ff', BLUE_S = '#2563eb', BLUE_T = '#1e40af';
        var GRAY_F = '#f9fafb', GRAY_S = '#bbb', GRAY_T = '#333';

        var now = 0, animId = null;
        var END_HOLD = 90;
        var DT = 3.2;
        // Commit lands a hair before the verify pass ends, so a paused step can show the
        // accept/reject marks and the newly committed tokens in the same frame.
        var COMMIT_LEAD = 0.6;

        // ---- Discrete beats: one per drafter step and per verification, plus start and end ----
        // Each beat freezes the clock at a moment where the interesting state is visible.
        var STEPS = [{
            t: 0,
            title: 'Nothing has run yet',
            body: 'Both sides start from the same prompt and must produce the same ' + OUT.length +
                  ' tokens. Left will spend one large-model pass per token. Right will draft ' + K +
                  ' guesses at a time and check them together.'
        }];
        (function buildSteps() {
            for (var r = 0; r < spec.rounds.length; r++) {
                var R = spec.rounds[r];
                for (var j = 0; j < K; j++) {
                    var seg = R.draftSegs[j];
                    STEPS.push({
                        // just past t1: a slot only counts as drafted once now >= its t1
                        t: seg.t1 + 0.01,
                        title: 'Round ' + (r + 1) + ', drafter step ' + (j + 1) + ' of ' + K,
                        body: 'The cheap guesser proposes "' + R.guesses[j] + '". This costs ' + T_DRAFT +
                              ' units against the large model\'s ' + T_TARGET +
                              ', and it is sequential: each guess feeds the next. Meanwhile the left side is ' +
                              'still grinding through single tokens.'
                    });
                }
                var rejected = R.accept < K;
                var body = 'One large-model pass scores all ' + K +
                           ' guesses at once. Reading left to right: ' + R.accept +
                           ' accepted' + (rejected ? ', then "' + R.guesses[R.accept] +
                           '" is rejected and everything after it is discarded' : '') + '. ';
                body += rejected
                    ? 'The pass also resamples a corrected token, "' + OUT[R.fromTarget] +
                      '", so the round still commits ' + (R.accept + 1) + ' tokens.'
                    : 'All ' + K + ' survived, so the pass also yields a free bonus token, "' +
                      OUT[R.fromTarget] + '": ' + (K + 1) + ' tokens from one pass.';
                STEPS.push({
                    t: R.verify.t1 - COMMIT_LEAD / 2,
                    title: 'Round ' + (r + 1) + ', verify' + (rejected ? '' : ' (best case)'),
                    body: body
                });
            }
            STEPS.push({
                t: vanilla.total + 40,
                title: 'Both sides have finished',
                body: 'Identical output. The left paid ' + OUT.length + ' large-model passes; the right paid ' +
                      spec.rounds.length + ', plus ' + (spec.rounds.length * K) +
                      ' cheap drafter steps and some wasted work on rejected guesses. Speculation did more ' +
                      'total work in less wall-clock time, because the work was arranged into fewer serial steps.'
            });
        })();
        var stepIdx = 0;

        function roundRect(x, y, w, h, r) {
            r = Math.min(r, w / 2, h / 2);
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.arcTo(x + w, y, x + w, y + h, r);
            ctx.arcTo(x + w, y + h, x, y + h, r);
            ctx.arcTo(x, y + h, x, y, r);
            ctx.arcTo(x, y, x + w, y, r);
            ctx.closePath();
        }

        function tokWidth(text) {
            ctx.font = '12px Georgia, serif';
            return Math.max(24, ctx.measureText(text).width + 16);
        }

        function drawToken(x, y, text, fill, stroke, txt, dashed, mark) {
            var w = tokWidth(text);
            ctx.save();
            if (dashed) ctx.setLineDash([3, 2]);
            roundRect(x, y, w, BOX_H, 5);
            ctx.fillStyle = fill;
            ctx.fill();
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();

            ctx.font = '12px Georgia, serif';
            ctx.fillStyle = txt;
            ctx.textAlign = 'center';
            ctx.fillText(text, x + w / 2, y + BOX_H / 2 + 4);

            if (mark === 'ok') {
                ctx.save();
                ctx.strokeStyle = GREEN_S;
                ctx.lineWidth = 1.8;
                ctx.beginPath();
                ctx.moveTo(x + w - 11, y - 5);
                ctx.lineTo(x + w - 8, y - 1);
                ctx.lineTo(x + w - 2, y - 10);
                ctx.stroke();
                ctx.restore();
            } else if (mark === 'no') {
                ctx.save();
                ctx.strokeStyle = RED_S;
                ctx.lineWidth = 1.8;
                ctx.beginPath();
                ctx.moveTo(x + w - 10, y - 9);
                ctx.lineTo(x + w - 2, y - 1);
                ctx.moveTo(x + w - 2, y - 9);
                ctx.lineTo(x + w - 10, y - 1);
                ctx.stroke();
                ctx.restore();
            }
            return w;
        }

        function label(x, y, text, size, color, align, italic, bold) {
            ctx.font = (italic ? 'italic ' : '') + (bold ? 'bold ' : '') + (size || 11) +
                       "px Georgia, serif";
            ctx.fillStyle = color || '#555';
            ctx.textAlign = align || 'left';
            ctx.fillText(text, x, y);
        }

        function drawTimeline(px, segs, upto, doneAt, showMark) {
            var x0 = px + TL_X_OFF;
            // track
            roundRect(x0, TL_Y, TL_W, TL_H, 4);
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 1;
            ctx.stroke();

            for (var i = 0; i < segs.length; i++) {
                var s = segs[i];
                if (s.t0 >= upto) break;
                var end = Math.min(s.t1, upto);
                var sx = x0 + s.t0 * PPT;
                var sw = Math.max(0.8, (end - s.t0) * PPT);
                ctx.fillStyle = s.kind === 'target' ? '#c7d7fb' : '#e6e6e2';
                ctx.fillRect(sx, TL_Y + 1, sw, TL_H - 2);
                if (s.kind === 'target' && sw > 3) {
                    ctx.strokeStyle = '#93b0f5';
                    ctx.lineWidth = 0.8;
                    ctx.strokeRect(sx + 0.4, TL_Y + 1.4, sw - 0.8, TL_H - 2.8);
                }
            }

            // finish marker
            if (doneAt !== null && upto >= doneAt) {
                var fx = x0 + doneAt * PPT;
                ctx.save();
                ctx.strokeStyle = GREEN_S;
                ctx.lineWidth = 1.6;
                ctx.beginPath();
                ctx.moveTo(fx, TL_Y - 6);
                ctx.lineTo(fx, TL_Y + TL_H + 6);
                ctx.stroke();
                ctx.restore();
                // flip the label inward when the marker lands near the panel's right edge
                var nearEnd = fx > x0 + TL_W * 0.6;
                label(fx + (nearEnd ? -5 : 5), TL_Y + TL_H + 18, 'done at ' + Math.round(doneAt),
                      10, GREEN_T, nearEnd ? 'right' : 'left', true);
            }

            // reference marker on the slow timeline
            if (showMark && upto >= spec.total) {
                var mx = x0 + spec.total * PPT;
                ctx.save();
                ctx.strokeStyle = '#d97706';
                ctx.lineWidth = 1.4;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(mx, TL_Y - 6);
                ctx.lineTo(mx, TL_Y + TL_H + 6);
                ctx.stroke();
                ctx.restore();
                label(mx + 5, TL_Y - 10, 'right side already finished', 10, '#92400e', 'left', true);
            }
        }

        function drawOutput(px, y, items) {
            var x = px + TL_X_OFF, line = 0;
            var maxX = px + TL_X_OFF + TL_W;
            for (var i = 0; i < items.length; i++) {
                var w = tokWidth(items[i].text);
                if (x + w > maxX) { line++; x = px + TL_X_OFF; }
                drawToken(x, y + line * LINE_H, items[i].text, items[i].f, items[i].s, items[i].t);
                x += w + 4;
            }
        }

        function activeSeg(segs, t) {
            for (var i = 0; i < segs.length; i++) {
                if (t >= segs[i].t0 && t < segs[i].t1) return segs[i];
            }
            return null;
        }

        function drawChips(px, draftOn, targetOn) {
            var y = 490;
            var x = px + TL_X_OFF;
            roundRect(x, y, 132, 26, 6);
            ctx.fillStyle = draftOn ? '#e6e6e2' : '#fff';
            ctx.fill();
            ctx.strokeStyle = draftOn ? '#888' : '#e8e8e8';
            ctx.lineWidth = draftOn ? 1.6 : 1;
            ctx.stroke();
            label(x + 66, y + 17, 'drafter running', 11, draftOn ? '#333' : '#ccc', 'center');

            roundRect(x + 146, y, 178, 26, 6);
            ctx.fillStyle = targetOn ? '#c7d7fb' : '#fff';
            ctx.fill();
            ctx.strokeStyle = targetOn ? BLUE_S : '#e8e8e8';
            ctx.lineWidth = targetOn ? 1.6 : 1;
            ctx.stroke();
            label(x + 235, y + 17, 'large model forward pass', 11, targetOn ? BLUE_T : '#ccc', 'center');
        }

        function render() {
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, W, H);

            // ---- header ----
            label(W / 2, 28, 'One prompt, one output, two decoding schedules', 17, '#1a1a1a', 'center', false, true);
            label(W / 2, 50, 'Large-model pass costs ' + T_TARGET + ' units. Drafter step costs ' + T_DRAFT +
                  '. Both timelines share one clock.', 12, '#555', 'center', true);

            // ---- panel frames ----
            [LX, RX].forEach(function (px) {
                roundRect(px, 72, PW, 452, 8);
                ctx.fillStyle = '#fdfdfd';
                ctx.fill();
                ctx.strokeStyle = '#e8e8e8';
                ctx.lineWidth = 1;
                ctx.stroke();
            });

            // ================= LEFT: vanilla =================
            label(LX + TL_X_OFF, 98, 'Plain autoregressive', 15, '#1a1a1a', 'left', false, true);
            label(LX + TL_X_OFF, 118, 'one large-model pass per token, ' + OUT.length + ' passes total',
                  11.5, '#555', 'left', true);
            label(LX + TL_X_OFF, 142, 'GPU time', 10.5, '#555');

            var vDone = now >= vanilla.total ? vanilla.total : null;
            drawTimeline(LX, vanilla.segs, now, vDone, true);

            var vSeg = activeSeg(vanilla.segs, now);
            var vEmitted = 0;
            for (var i = 0; i < vanilla.segs.length; i++) {
                if (now >= vanilla.segs[i].t1) vEmitted = i + 1;
            }
            label(LX + TL_X_OFF, 224, 'large-model passes: ' + Math.min(OUT.length, Math.ceil(now / T_TARGET)) +
                  '     tokens out: ' + vEmitted, 11.5, '#333');

            label(LX + TL_X_OFF, 252, now > 0 ? 'this pass produces' : ' ', 10.5, '#555');
            if (vSeg && now > 0) {
                var pending = OUT[vSeg.emits];
                var frac = (now - vSeg.t0) / (vSeg.t1 - vSeg.t0);
                var bw = tokWidth(pending);
                // progress fill behind the token
                ctx.fillStyle = '#eaf0fe';
                ctx.fillRect(LX + TL_X_OFF, PASS_Y, bw * frac, PASS_H);
                drawToken(LX + TL_X_OFF, PASS_Y, pending, 'rgba(0,0,0,0)', BLUE_S, BLUE_T);
                label(LX + TL_X_OFF + bw + 12, PASS_Y + 17, 'exactly one token', 11, '#555', 'left', true);
            } else if (now >= vanilla.total) {
                label(LX + TL_X_OFF, PASS_Y + 17, 'finished', 11, GREEN_T, 'left', true);
            }

            label(LX + TL_X_OFF, 320, 'committed output', 10.5, '#555');
            var vItems = [];
            for (var i = 0; i < vEmitted; i++) {
                vItems.push({ text: OUT[i], f: BLUE_F, s: BLUE_S, t: BLUE_T });
            }
            drawOutput(LX, OUTQ_Y, vItems);
            drawChips(LX, false, !!vSeg && now > 0);

            // ================= RIGHT: speculative =================
            label(RX + TL_X_OFF, 98, 'Speculative, k = ' + K, 15, '#1a1a1a', 'left', false, true);
            label(RX + TL_X_OFF, 118, 'draft ' + K + ' guesses, verify them in one pass', 11.5, '#555', 'left', true);
            label(RX + TL_X_OFF, 142, 'GPU time', 10.5, '#555');

            var sDone = now >= spec.total ? spec.total : null;
            drawTimeline(RX, spec.segs, now, sDone, false);

            var sSeg = activeSeg(spec.segs, now);
            var passCount = 0, sCommitted = [];
            for (var r = 0; r < spec.rounds.length; r++) {
                var R = spec.rounds[r];
                if (now >= R.verify.t1 - COMMIT_LEAD) {
                    passCount++;
                    for (var a = 0; a < R.fromDraft.length; a++) {
                        sCommitted.push({ text: OUT[R.fromDraft[a]], f: GREEN_F, s: GREEN_S, t: GREEN_T });
                    }
                    sCommitted.push({ text: OUT[R.fromTarget], f: BLUE_F, s: BLUE_S, t: BLUE_T });
                }
            }
            var draftStepsDone = 0;
            for (var i = 0; i < spec.segs.length; i++) {
                if (spec.segs[i].kind === 'draft' && now >= spec.segs[i].t1) draftStepsDone++;
            }
            label(RX + TL_X_OFF, 224, 'large-model passes: ' + passCount + '     drafter steps: ' +
                  draftStepsDone + '     tokens out: ' + sCommitted.length, 11.5, '#333');

            // current round's speculation buffer
            var curRound = null;
            if (sSeg) curRound = spec.rounds[sSeg.round];
            else if (now < spec.total) curRound = spec.rounds[0];

            if (curRound && now < spec.total) {
                var verifying = sSeg && sSeg.kind === 'target';
                var vfrac = verifying ? (now - sSeg.t0) / (sSeg.t1 - sSeg.t0) : 0;
                // At the exact instant verification begins, the honest caption is that drafting
                // just finished: the sweep has not moved yet.
                var swept = verifying && vfrac > 0.02;
                var hint = now === 0 ? ' '
                         : swept ? 'this pass verifies all ' + K + ' guesses at once'
                         : verifying ? 'all ' + K + ' guesses drafted, verification about to run'
                         : 'drafting guesses (cheap, sequential)';
                label(RX + TL_X_OFF, 252, hint, 10.5, swept ? BLUE_T : '#555');

                var gx = RX + TL_X_OFF;
                for (var j = 0; j < K; j++) {
                    var g = curRound.guesses[j];
                    var gw = tokWidth(g);
                    var drafted = now >= curRound.draftSegs[j].t1;
                    var resolved = verifying && vfrac > (j + 1) / (K + 1);
                    if (!drafted) {
                        // slot not yet filled
                        ctx.save();
                        ctx.setLineDash([3, 2]);
                        roundRect(gx, PASS_Y, gw, PASS_H, 5);
                        ctx.strokeStyle = '#ddd';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                        ctx.restore();
                    } else if (!resolved) {
                        drawToken(gx, PASS_Y, g, GRAY_F, GRAY_S, GRAY_T, true);
                    } else if (j < curRound.accept) {
                        drawToken(gx, PASS_Y, g, GREEN_F, GREEN_S, GREEN_T, false, 'ok');
                    } else if (j === curRound.accept) {
                        drawToken(gx, PASS_Y, g, RED_F, RED_S, RED_T, false, 'no');
                    } else {
                        ctx.globalAlpha = 0.35;
                        drawToken(gx, PASS_Y, g, GRAY_F, GRAY_S, GRAY_T, true);
                        ctx.globalAlpha = 1;
                    }
                    gx += gw + 4;
                }

                // the token the target itself contributes
                var tailText = OUT[curRound.fromTarget];
                var tw = tokWidth(tailText);
                if (verifying && vfrac > 0.92) {
                    drawToken(gx + 10, PASS_Y, tailText, BLUE_F, BLUE_S, BLUE_T);
                    label(gx + 10, PASS_Y + PASS_H + 14, curRound.bonus ? 'bonus token' : 'resampled here',
                          10, BLUE_T, 'left', true);
                } else {
                    ctx.save();
                    ctx.setLineDash([3, 2]);
                    roundRect(gx + 10, PASS_Y, tw, PASS_H, 5);
                    ctx.strokeStyle = '#e4e4e4';
                    ctx.stroke();
                    ctx.restore();
                }

                // verification sweep
                if (verifying) {
                    var sweepW = (gx + 10 + tw - (RX + TL_X_OFF)) * Math.min(1, vfrac);
                    ctx.save();
                    ctx.globalAlpha = 0.16;
                    ctx.fillStyle = BLUE_S;
                    ctx.fillRect(RX + TL_X_OFF, PASS_Y - 4, sweepW, PASS_H + 8);
                    ctx.restore();
                }
            } else if (now >= spec.total) {
                label(RX + TL_X_OFF, 252, ' ', 10.5, '#555');
                label(RX + TL_X_OFF, PASS_Y + 17, 'finished', 11, GREEN_T, 'left', true);
            }

            label(RX + TL_X_OFF, 320, 'committed output', 10.5, '#555');
            drawOutput(RX, OUTQ_Y, sCommitted);
            drawChips(RX, !!(sSeg && sSeg.kind === 'draft') && now > 0,
                          !!(sSeg && sSeg.kind === 'target'));

            // ---- legend ----
            label(LX + TL_X_OFF, 540, 'green = written by the drafter, approved by the large model', 11.5, GREEN_T);
            label(RX + TL_X_OFF, 540, 'blue = sampled directly by the large model', 11.5, BLUE_T);

            // ---- verdict ----
            if (now >= vanilla.total) {
                var ratio = (vanilla.total / spec.total).toFixed(2);
                label(W / 2, 566, 'Identical ' + OUT.length + ' tokens, ' + ratio + 'x less wall-clock time.',
                      14, '#1a1a1a', 'center', false, true);
                label(W / 2, 586, OUT.length + ' large-model passes (' + vanilla.total + ' units) versus ' +
                      spec.rounds.length + ' passes plus ' + (spec.rounds.length * K) + ' drafter steps (' +
                      spec.total + ' units).', 12, '#555', 'center', true);
            }
        }

        function stop() {
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            playBtn.textContent = 'Play';
        }

        function showStep(i) {
            stepIdx = Math.max(0, Math.min(STEPS.length - 1, i));
            var s = STEPS[stepIdx];
            now = s.t;
            render();
            caption.innerHTML = '<strong>' + s.title + '.</strong> ' + s.body;
            countEl.textContent = 'step ' + stepIdx + ' of ' + (STEPS.length - 1);
            prevBtn.disabled = stepIdx === 0;
            nextBtn.disabled = stepIdx === STEPS.length - 1;
            prevBtn.style.opacity = prevBtn.disabled ? 0.4 : 1;
            nextBtn.style.opacity = nextBtn.disabled ? 0.4 : 1;
        }

        function tick() {
            now += DT;
            render();
            // keep the caption in sync with whichever beat the clock has passed
            var i = 0;
            for (var k = 0; k < STEPS.length; k++) if (now >= STEPS[k].t) i = k;
            if (i !== stepIdx) {
                stepIdx = i;
                caption.innerHTML = '<strong>' + STEPS[i].title + '.</strong> ' + STEPS[i].body;
                countEl.textContent = 'step ' + i + ' of ' + (STEPS.length - 1);
            }
            if (now >= vanilla.total + END_HOLD) {
                animId = null;
                showStep(STEPS.length - 1);
                return;
            }
            animId = requestAnimationFrame(tick);
        }

        nextBtn.addEventListener('click', function () { stop(); showStep(stepIdx + 1); });
        prevBtn.addEventListener('click', function () { stop(); showStep(stepIdx - 1); });
        resetBtn.addEventListener('click', function () { stop(); showStep(0); });

        playBtn.addEventListener('click', function () {
            if (animId) { stop(); return; }
            // replay from the top if we are already at the end
            if (now >= vanilla.total) showStep(0);
            playBtn.textContent = 'Pause';
            prevBtn.disabled = nextBtn.disabled = false;
            prevBtn.style.opacity = nextBtn.style.opacity = 1;
            animId = requestAnimationFrame(tick);
        });

        showStep(0);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mount);
    } else {
        mount();
    }
})();
