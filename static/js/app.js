(() => {
    const LOGICAL_SIZE = 280;
    const CANVAS_SCALE = 2; // 560x560 backing buffer for crisp HiDPI strokes

    const canvas = document.getElementById("digit-canvas");
    const ctx = canvas.getContext("2d", { willReadFrequently: false });

    const brushSlider = document.getElementById("brush-size");
    const brushValueEl = document.getElementById("brush-value");
    const brushPreviewDot = document.getElementById("brush-preview-dot");
    const realtimeToggle = document.getElementById("realtime-toggle");
    const modeLabel = document.getElementById("mode-label");

    const undoBtn = document.getElementById("undo-btn");
    const clearBtn = document.getElementById("clear-btn");
    const predictBtn = document.getElementById("predict-btn");

    const statusBadge = document.getElementById("status-badge");
    const predictedDigitEl = document.getElementById("predicted-digit");
    const confidenceValueEl = document.getElementById("confidence-value");
    const confidenceBarEl = document.getElementById("confidence-bar");
    const probabilityListEl = document.getElementById("probability-list");

    // Offscreen hi-res backing setup
    canvas.width = LOGICAL_SIZE * CANVAS_SCALE;
    canvas.height = LOGICAL_SIZE * CANVAS_SCALE;
    ctx.setTransform(CANVAS_SCALE, 0, 0, CANVAS_SCALE, 0, 0);

    let brushSize = Number(brushSlider.value) || 22;
    let isDrawing = false;
    let strokes = [];
    let currentStroke = null;
    let realtimeDebounceTimer = null;
    let activeController = null;
    let apiBase = "";

    // Build 0-9 probability rows
    const probRows = [];
    function initProbabilityRows() {
        probabilityListEl.innerHTML = "";
        for (let digit = 0; digit <= 9; digit++) {
            const row = document.createElement("div");
            row.className = "prob-row";

            const badge = document.createElement("span");
            badge.className = "prob-digit";
            badge.textContent = String(digit);

            const track = document.createElement("div");
            track.className = "prob-bar-track";

            const fill = document.createElement("div");
            fill.className = "prob-bar-fill";
            track.appendChild(fill);

            const pct = document.createElement("span");
            pct.className = "prob-pct";
            pct.textContent = "0.0%";

            row.append(badge, track, pct);
            probabilityListEl.appendChild(row);
            probRows.push({ row, fill, pct });
        }
    }

    function setStatus(text, variant = "ready") {
        statusBadge.textContent = text;
        statusBadge.classList.remove("status-busy", "status-error");
        if (variant === "busy") {
            statusBadge.classList.add("status-busy");
        } else if (variant === "error") {
            statusBadge.classList.add("status-error");
        }
    }

    function resetOutput() {
        predictedDigitEl.textContent = "—";
        confidenceValueEl.textContent = "0.0%";
        confidenceBarEl.style.width = "0%";
        probRows.forEach(({ row, fill, pct }) => {
            row.classList.remove("is-top");
            fill.style.width = "0%";
            pct.textContent = "0.0%";
        });
    }

    function renderPrediction(data) {
        if (!data || data.blank) {
            resetOutput();
            return;
        }

        const predicted = Number(data.predicted_digit);
        const confidencePct = Number(data.confidence) * 100;
        const probs = Array.isArray(data.probabilities) ? data.probabilities : [];

        predictedDigitEl.textContent = String(predicted);
        confidenceValueEl.textContent = `${confidencePct.toFixed(1)}%`;
        confidenceBarEl.style.width = `${Math.min(100, Math.max(0, confidencePct)).toFixed(1)}%`;

        probRows.forEach(({ row, fill, pct }, digit) => {
            const val = Number(probs[digit] ?? 0) * 100;
            const clamped = Math.min(100, Math.max(0, val));
            fill.style.width = `${clamped.toFixed(1)}%`;
            pct.textContent = `${clamped.toFixed(1)}%`;
            row.classList.toggle("is-top", digit === predicted);
        });
    }

    function clearCanvasPixels() {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, LOGICAL_SIZE, LOGICAL_SIZE);
    }

    function drawSingleStroke(stroke) {
        if (!stroke || stroke.points.length === 0) return;

        ctx.save();
        ctx.strokeStyle = "#000000";
        ctx.fillStyle = "#000000";
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = stroke.size;

        const pts = stroke.points;
        if (pts.length === 1) {
            ctx.beginPath();
            ctx.arc(pts[0].x, pts[0].y, stroke.size / 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
            return;
        }

        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length - 1; i++) {
            const midX = (pts[i].x + pts[i + 1].x) / 2;
            const midY = (pts[i].y + pts[i + 1].y) / 2;
            ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
        }
        const last = pts[pts.length - 1];
        ctx.lineTo(last.x, last.y);
        ctx.stroke();
        ctx.restore();
    }

    function redrawAllStrokes() {
        clearCanvasPixels();
        for (const stroke of strokes) {
            drawSingleStroke(stroke);
        }
        undoBtn.disabled = strokes.length === 0;
    }

    function updateBrushUI() {
        brushSize = Number(brushSlider.value);
        brushValueEl.textContent = `${brushSize}px`;

        const min = Number(brushSlider.min);
        const max = Number(brushSlider.max);
        const pct = ((brushSize - min) / (max - min)) * 100;
        brushSlider.style.setProperty("--slider-pct", `${pct}%`);

        const dotSize = Math.max(5, Math.min(22, Math.round(brushSize * 0.52)));
        brushPreviewDot.style.width = `${dotSize}px`;
        brushPreviewDot.style.height = `${dotSize}px`;
    }

    function getCanvasPoint(event) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = LOGICAL_SIZE / rect.width;
        const scaleY = LOGICAL_SIZE / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY,
        };
    }

    async function resolveBackend() {
        const candidates = [""];
        if (window.location.port !== "8000") {
            candidates.push("http://127.0.0.1:8000", "http://localhost:8000");
        }

        for (const base of candidates) {
            try {
                const res = await fetch(`${base}/health`);
                if (!res.ok) continue;
                const ct = res.headers.get("content-type") || "";
                if (!ct.includes("application/json")) continue;
                const info = await res.json();
                if (typeof info.model_loaded !== "boolean") continue;

                apiBase = base;
                return info;
            } catch {
                // Try next backend candidate
            }
        }
        return null;
    }

    async function runPrediction() {
        if (strokes.length === 0) {
            resetOutput();
            setStatus("AI Ready", "ready");
            return;
        }

        if (activeController) {
            activeController.abort();
        }
        activeController = new AbortController();

        setStatus("Predicting...", "busy");

        try {
            const imageDataUrl = canvas.toDataURL("image/png");
            let response = await fetch(`${apiBase}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: imageDataUrl }),
                signal: activeController.signal,
            });

            if (!response.ok && window.location.port !== "8000") {
                const info = await resolveBackend();
                if (info && info.model_loaded) {
                    response = await fetch(`${apiBase}/predict`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ image: imageDataUrl }),
                        signal: activeController.signal,
                    });
                }
            }

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            renderPrediction(data);
            setStatus("AI Ready", "ready");
        } catch (err) {
            if (err.name === "AbortError") {
                return;
            }
            console.error("Prediction error:", err);
            setStatus("Offline", "error");
            scheduleHealthRetry();
        }
    }

    function scheduleRealtimePrediction(immediate = false) {
        if (!realtimeToggle.checked) return;

        clearTimeout(realtimeDebounceTimer);
        if (immediate) {
            runPrediction();
            return;
        }
        realtimeDebounceTimer = setTimeout(() => {
            runPrediction();
        }, 120);
    }

    function startStroke(event) {
        if (event.button !== undefined && event.button !== 0 && event.pointerType === "mouse") {
            return;
        }
        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);

        isDrawing = true;
        const pt = getCanvasPoint(event);
        currentStroke = { size: brushSize, points: [pt] };
        strokes.push(currentStroke);
        redrawAllStrokes();
        scheduleRealtimePrediction(false);
    }

    function continueStroke(event) {
        if (!isDrawing || !currentStroke) return;
        event.preventDefault();
        const pt = getCanvasPoint(event);
        const pts = currentStroke.points;
        const prev = pts[pts.length - 1];
        const dx = pt.x - prev.x;
        const dy = pt.y - prev.y;
        if (dx * dx + dy * dy < 1.5) return;

        pts.push(pt);
        redrawAllStrokes();
        scheduleRealtimePrediction(false);
    }

    function endStroke(event) {
        if (!isDrawing) return;
        isDrawing = false;
        currentStroke = null;
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
        scheduleRealtimePrediction(true);
    }

    function handleUndo() {
        if (strokes.length === 0) return;
        clearTimeout(realtimeDebounceTimer);
        strokes.pop();
        redrawAllStrokes();

        if (strokes.length === 0) {
            if (activeController) activeController.abort();
            resetOutput();
            setStatus("AI Ready", "ready");
        } else if (realtimeToggle.checked) {
            runPrediction();
        }
    }

    function handleClear() {
        clearTimeout(realtimeDebounceTimer);
        if (activeController) activeController.abort();
        strokes = [];
        currentStroke = null;
        isDrawing = false;
        redrawAllStrokes();
        resetOutput();
        setStatus("AI Ready", "ready");
    }

    let healthRetryTimer = null;

    function scheduleHealthRetry() {
        if (healthRetryTimer) return;
        healthRetryTimer = setTimeout(() => {
            healthRetryTimer = null;
            checkBackendHealth();
        }, 2500);
    }

    async function checkBackendHealth() {
        const info = await resolveBackend();
        if (info) {
            setStatus(info.model_loaded ? "AI Ready" : "Model Missing", info.model_loaded ? "ready" : "error");
            if (info.model_loaded && strokes.length > 0 && predictedDigitEl.textContent === "—") {
                runPrediction();
            }
            return;
        }
        setStatus("Offline", "error");
        scheduleHealthRetry();
    }

    // Event bindings
    brushSlider.addEventListener("input", updateBrushUI);

    realtimeToggle.addEventListener("change", () => {
        modeLabel.textContent = realtimeToggle.checked ? "Realtime" : "Manual";
        if (realtimeToggle.checked && strokes.length > 0) {
            runPrediction();
        }
    });

    undoBtn.addEventListener("click", handleUndo);
    clearBtn.addEventListener("click", handleClear);
    predictBtn.addEventListener("click", runPrediction);

    canvas.addEventListener("pointerdown", startStroke);
    canvas.addEventListener("pointermove", continueStroke);
    canvas.addEventListener("pointerup", endStroke);
    canvas.addEventListener("pointercancel", endStroke);

    window.addEventListener("keydown", (event) => {
        const tag = event.target?.tagName?.toLowerCase();
        if (tag === "input" && event.target.type !== "range" && event.target.type !== "checkbox") {
            return;
        }
        const key = event.key.toLowerCase();
        const hasMod = event.ctrlKey || event.metaKey || event.altKey;
        if (key === "c" && !hasMod) {
            event.preventDefault();
            handleClear();
        } else if (key === "z" && !event.altKey) {
            event.preventDefault();
            handleUndo();
        } else if (event.key === "Enter" && !hasMod && tag !== "button") {
            event.preventDefault();
            runPrediction();
        }
    });

    // Initialize
    initProbabilityRows();
    updateBrushUI();
    redrawAllStrokes();
    resetOutput();
    checkBackendHealth();
})();
