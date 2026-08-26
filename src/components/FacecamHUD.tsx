import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DrawingUtils,
  FaceDetector,
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import MilyonusMark from "@/components/MilyonusMark";
import { poseFromTransformMatrix, type HeadPose } from "@/lib/facePose";
import { FaceTracker } from "@/lib/faceTracker";
import {
  EMOTIONS,
  dominantEmotion,
  emptyScores,
  scoresFromBlendshapes,
  type Emotion,
  type Scores,
} from "@/lib/blendshapeEmotion";
import {
  buildMinuteReports,
  focusScoreFromPose,
  gazeDirectionFromPose,
  samplesToCsv,
  type GazeDirection,
  type SecondSample,
} from "@/lib/focusTracking";
import { ParameterEngine } from "@/lib/parameterEngine";
import { calculateDeepTechScore } from "@/lib/deepTech";
import { recorder } from "@/lib/recorder";
import { LandmarkProcessor } from "@/lib/landmarkProcessor";

const TASKS_VISION_VERSION = "1.0.1";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;
const FACE_LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const FACE_DETECTOR_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite";

const MAX_FACES = 4;
const NOSE_TIP_INDEX = 1;

const EMOTION_TOKEN: Record<Emotion, string> = {
  smile: "var(--happy)",
  sad: "var(--sad)",
  angry: "var(--angry)",
  surprised: "var(--surprise)",
  fearful: "var(--fear)",
  disgusted: "var(--happy)",
  thinking: "var(--thinking)",
  neutral: "var(--brand-bright)",
};

const GAZE_LABEL: Record<GazeDirection, string> = {
  center: "ON SCREEN",
  left: "LOOKING LEFT",
  right: "LOOKING RIGHT",
  up: "LOOKING UP",
  down: "LOOKING DOWN",
  none: "NO FACE",
};

const GAZE_COLOR: Record<GazeDirection, string> = {
  center: "var(--happy)",
  left: "var(--angry)",
  right: "var(--angry)",
  up: "var(--thinking)",
  down: "var(--thinking)",
  none: "var(--sad)",
};

const vpe = new ParameterEngine();

type LogEntry = { t: string; msg: string; emotion: Emotion };

type PixelBox = { x: number; y: number; width: number; height: number };

interface FaceReading {
  id: number;
  confidence: number | null;
  pose: HeadPose;
  scores: Scores;
  dominant: Emotion;
  box: PixelBox;
  area: number;
  landmarkCount: number;
}

function cssVar(name: string) {
  if (typeof window === "undefined") return "#3b82f6";
  const raw = name.replace("var(", "").replace(")", "");
  return getComputedStyle(document.documentElement).getPropertyValue(raw).trim() || "#3b82f6";
}

function stamp(d = new Date()) {
  return d.toTimeString().slice(0, 8) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

function boxFromLandmarks(pts: NormalizedLandmark[], vw: number, vh: number): PixelBox {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    const x = p.x * vw;
    const y = p.y * vh;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function drawTargetBox(ctx: CanvasRenderingContext2D, box: PixelBox, color: string) {
  const { x, y, width, height } = box;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 6;
  const c = Math.min(width, height) * 0.22;
  const corners: Array<[number, number, number, number]> = [
    [x, y, 1, 1],
    [x + width, y, -1, 1],
    [x, y + height, 1, -1],
    [x + width, y + height, -1, -1],
  ];
  corners.forEach(([cx, cy, sx, sy]) => {
    ctx.beginPath();
    ctx.moveTo(cx, cy + sy * c);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + sx * c, cy);
    ctx.stroke();
  });
  ctx.shadowBlur = 0;
}

/** Draws an RGB axis gizmo at the nose tip from the raw column-major rotation matrix. */
function drawPoseGizmo(
  ctx: CanvasRenderingContext2D,
  origin: { x: number; y: number },
  matrix: readonly number[],
  length: number,
) {
  const axes: Array<{ vec: [number, number]; color: string }> = [
    { vec: [matrix[0]!, matrix[1]!], color: "#ff4d6d" }, // local X
    { vec: [matrix[4]!, matrix[5]!], color: "#4dff88" }, // local Y
    { vec: [matrix[8]!, matrix[9]!], color: "#4d9bff" }, // local Z (facing)
  ];
  ctx.lineWidth = 2.2;
  axes.forEach(({ vec, color }) => {
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(origin.x + vec[0] * length, origin.y - vec[1] * length);
    ctx.stroke();
  });
  ctx.shadowBlur = 0;
}

function drawFaceLabel(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  box: PixelBox,
  id: number,
  confidence: number | null,
  color: string,
) {
  const mirroredCenterX = canvasWidth - (box.x + box.width / 2);
  const text = `ID ${id} · ${confidence == null ? "--" : Math.round(confidence * 100) + "%"}`;
  ctx.font = "600 13px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  const padY = box.y - 10;
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  const w = ctx.measureText(text).width + 14;
  ctx.fillRect(mirroredCenterX - w / 2, padY - 16, w, 20);
  ctx.fillStyle = color;
  ctx.fillText(text, mirroredCenterX, padY);
}

export default function FacecamHUD() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const lastCalloutRef = useRef<{ faceId: number; e: Emotion | null; at: number }>({
    faceId: -1,
    e: null,
    at: 0,
  });
  const fpsRef = useRef<{ last: number; frames: number }>({ last: 0, frames: 0 });
  const mutedRef = useRef(false);
  const trackerRef = useRef(new FaceTracker());
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const detectorRef = useRef<FaceDetector | null>(null);
  const drawingUtilsRef = useRef<DrawingUtils | null>(null);
  const processorRef = useRef(new LandmarkProcessor());
  const lastFaceCountRef = useRef(0);
  const latestPrimaryRef = useRef<{ confidence: number | null; pose: HeadPose | null; deepTechScore: number | null }>({
    confidence: null,
    pose: null,
    deepTechScore: null,
  });
  const sessionStartRef = useRef(0);
  const lastGazeAwayLogRef = useRef(0);

  const [phase, setPhase] = useState<"gate" | "loading" | "live" | "error">("gate");
  const [errorMsg, setErrorMsg] = useState("");
  const [faces, setFaces] = useState<FaceReading[]>([]);
  const [callout, setCallout] = useState<{ e: Emotion; id: number } | null>(null);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [muted, setMuted] = useState(false);
  const [fps, setFps] = useState(0);
  const [clock, setClock] = useState("--:--:--");
  const [samples, setSamples] = useState<SecondSample[]>([]);

  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  useEffect(() => {
    const i = setInterval(() => setClock(new Date().toTimeString().slice(0, 8)), 1000);
    return () => clearInterval(i);
  }, []);

  const blip = useCallback((emotion: Emotion) => {
    if (mutedRef.current) return;
    try {
      const ctx =
        audioCtxRef.current ??
        (audioCtxRef.current = new (
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
        )());
      const base = 320 + EMOTIONS.indexOf(emotion) * 90;
      [0, 0.08].forEach((offset, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "square";
        osc.frequency.value = base * (idx + 1);
        gain.gain.setValueAtTime(0.0001, ctx.currentTime + offset);
        gain.gain.exponentialRampToValueAtTime(0.09, ctx.currentTime + offset + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + offset + 0.09);
        osc.connect(gain).connect(ctx.destination);
        osc.start(ctx.currentTime + offset);
        osc.stop(ctx.currentTime + offset + 0.1);
      });
    } catch {
      /* audio unavailable */
    }
  }, []);

  const pushLog = useCallback((id: number, emotion: Emotion, conf: number) => {
    setLog((prev) =>
      [
        {
          t: stamp(),
          emotion,
          msg: `Subject ${id}: ${emotion} detected — confidence ${(conf * 100).toFixed(1)}%`,
        },
        ...prev,
      ].slice(0, 80),
    );
  }, []);

  const pushSystemLog = useCallback((msg: string) => {
    setLog((prev) => [{ t: stamp(), emotion: "neutral" as Emotion, msg }, ...prev].slice(0, 80));
  }, []);

  const loop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    const detector = detectorRef.current;

    if (!video || !canvas || !landmarker || !detector || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(() => loop());
      return;
    }

    const now = Math.round(performance.now());
    const landmarkResult = landmarker.detectForVideo(video, now);
    const detectResult = detector.detectForVideo(video, now);

    fpsRef.current.frames += 1;
    if (now - fpsRef.current.last > 1000) {
      setFps(fpsRef.current.frames);
      fpsRef.current.frames = 0;
      fpsRef.current.last = now;
    }

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw;
      canvas.height = vh;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      rafRef.current = requestAnimationFrame(() => loop());
      return;
    }
    if (!drawingUtilsRef.current) drawingUtilsRef.current = new DrawingUtils(ctx);
    const drawingUtils = drawingUtilsRef.current;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const faceLandmarks = landmarkResult.faceLandmarks;
    const rawBoxes = faceLandmarks.map((pts) => boxFromLandmarks(pts, vw, vh));

    const trackIds = trackerRef.current.update(
      rawBoxes.map((b) => ({
        cx: b.x + b.width / 2,
        cy: b.y + b.height / 2,
        w: b.width,
        h: b.height,
      })),
      now,
    );

    const smoothedLandmarks = faceLandmarks.map((pts, i) =>
      processorRef.current.smooth(trackIds[i], pts)
    );
    const boxes = smoothedLandmarks.map((pts) => boxFromLandmarks(pts, vw, vh));


    // Pair each landmark face with the nearest FaceDetector detection to read
    // its real detection-confidence score.
    const detections = detectResult.detections;
    const usedDet = new Set<number>();
    const confidences: Array<number | null> = boxes.map((box) => {
      const bcx = box.x + box.width / 2;
      const bcy = box.y + box.height / 2;
      let bestI = -1;
      let bestDist = Infinity;
      detections.forEach((d, i) => {
        if (usedDet.has(i) || !d.boundingBox) return;
        const dcx = d.boundingBox.originX + d.boundingBox.width / 2;
        const dcy = d.boundingBox.originY + d.boundingBox.height / 2;
        const dist = Math.hypot(bcx - dcx, bcy - dcy);
        if (dist < bestDist) {
          bestDist = dist;
          bestI = i;
        }
      });
      if (bestI === -1) return null;
      usedDet.add(bestI);
      return detections[bestI]!.categories[0]?.score ?? null;
    });

    const trackIds = trackerRef.current.update(
      boxes.map((b) => ({
        cx: b.x + b.width / 2,
        cy: b.y + b.height / 2,
        w: b.width,
        h: b.height,
      })),
      now,
    );

    // Mirror the mesh/graphics layer to match the mirrored <video>, then draw
    // text labels unflipped afterwards so they stay readable.
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    const readings: FaceReading[] = [];
    for (let i = 0; i < faceLandmarks.length; i++) {
      const pts = faceLandmarks[i]!;
      const box = boxes[i]!;
      const matrix = landmarkResult.facialTransformationMatrixes[i]?.data;
      const pose = matrix
        ? poseFromTransformMatrix(matrix)
        : { yaw: 0, pitch: 0, roll: 0, distance: 0 };
      const categories = landmarkResult.faceBlendshapes[i]?.categories ?? [];
      const scores = scoresFromBlendshapes(categories, pose.roll);
      const dominant = dominantEmotion(scores);
      const id = trackIds[i]!;
      const color = cssVar(EMOTION_TOKEN[dominant]);

      drawingUtils.drawConnectors(pts, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
        color: `${color}33`,
        lineWidth: 0.5,
      });
      drawingUtils.drawConnectors(pts, FaceLandmarker.FACE_LANDMARKS_CONTOURS, {
        color,
        lineWidth: 1.4,
      });
      drawingUtils.drawConnectors(pts, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, {
        color,
        lineWidth: 1.6,
      });
      drawingUtils.drawConnectors(pts, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, {
        color,
        lineWidth: 1.6,
      });

      // Precision Markers: Iris and Mouth Corners
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      [468, 473, 61, 291].forEach(idx => {
        const p = pts[idx];
        ctx.beginPath();
        ctx.arc(p.x * vw, p.y * vh, 2, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.shadowBlur = 0;

      drawTargetBox(ctx, box, color);
      if (matrix) {
        const nose = pts[NOSE_TIP_INDEX]!;
        drawPoseGizmo(
          ctx,
          { x: nose.x * vw, y: nose.y * vh },
          matrix,
          Math.max(30, box.width * 0.35),
        );
      }

      readings.push({
        id,
        confidence: confidences[i] ?? null,
        pose,
        scores,
        dominant,
        box,
        area: box.width * box.height,
        landmarkCount: pts.length,
      });
    }
    ctx.restore();

    for (let i = 0; i < readings.length; i++) {
      const r = readings[i]!;
      drawFaceLabel(
        ctx,
        canvas.width,
        r.box,
        r.id,
        r.confidence,
        cssVar(EMOTION_TOKEN[r.dominant]),
      );
    }

    readings.sort((a, b) => a.id - b.id);
    setFaces(readings);

    if (readings.length !== lastFaceCountRef.current) {
      pushSystemLog(
        readings.length === 0
          ? "No face in frame"
          : `Face count changed — now tracking ${readings.length} face${readings.length === 1 ? "" : "s"}`,
      );
      lastFaceCountRef.current = readings.length;
    }

    const primary = readings.reduce<FaceReading | null>(
      (best, r) => (!best || r.area > best.area ? r : best),
      null,
    );
    const deepTechScore = primary
      ? calculateDeepTechScore(primary.pose, primary.scores, primary.area, vpe)
      : null;
    latestPrimaryRef.current = {
      confidence: primary?.confidence ?? null,
      pose: primary?.pose ?? null,
      deepTechScore,
    };

    // Background High-Res Recording (Every Frame)
    if (primary) {
      recorder.record({
        id: primary.id,
        confidence: primary.confidence,
        pose: primary.pose,
        scores: primary.scores,
        dominant: primary.dominant,
        area: primary.area,
        deepTechScore,
      });
    }

    if (primary) {
      const conf = primary.scores[primary.dominant];
      const last = lastCalloutRef.current;
      if (
        primary.dominant !== "neutral" &&
        conf > 0.72 &&
        (last.e !== primary.dominant || last.faceId !== primary.id || now - last.at > 3200)
      ) {
        lastCalloutRef.current = { e: primary.dominant, faceId: primary.id, at: now };
        setCallout({ e: primary.dominant, id: now });
        pushLog(primary.id, primary.dominant, conf);
        blip(primary.dominant);
      }
    }

    rafRef.current = requestAnimationFrame(() => loop());
  }, [blip, pushLog, pushSystemLog]);

  useEffect(() => {
    if (!callout) return;
    const t = setTimeout(() => setCallout(null), 2000);
    return () => clearTimeout(t);
  }, [callout]);

  // Records one confidence + focus/gaze sample per second, independent of the
  // per-frame render loop, so the history is a clean 1Hz time series.
  useEffect(() => {
    if (phase !== "live") return;
    const id = setInterval(() => {
      const { confidence, pose, deepTechScore } = latestPrimaryRef.current;
      const gaze = gazeDirectionFromPose(pose);
      const focusScore = focusScoreFromPose(pose);
      const tSec = Math.round((Date.now() - sessionStartRef.current) / 1000);
      const sample: SecondSample = {
        tSec,
        atISO: new Date().toISOString(),
        hasFace: pose != null,
        confidence,
        yaw: pose?.yaw ?? null,
        pitch: pose?.pitch ?? null,
        distance: pose?.distance ?? null,
        area: faces.length > 0
          ? faces.reduce((best, r) => r.area > (best?.area ?? -1) ? r : best, null)?.area ?? null
          : null,
        deepTechScore,
        gaze,
        focusScore,
      };
      setSamples((prev) => [...prev, sample].slice(-14400));

      if (gaze === "left" || gaze === "right") {
        const now = Date.now();
        if (now - lastGazeAwayLogRef.current > 4000) {
          lastGazeAwayLogRef.current = now;
          pushSystemLog(
            `Focus alert — candidate looking ${gaze} (yaw ${pose?.yaw.toFixed(1)}°), confidence ${
              confidence != null ? `${Math.round(confidence * 100)}%` : "--"
            }`,
          );
        }
      }
    }, 1000);
    return () => clearInterval(id);
  }, [phase, pushSystemLog]);

  const minuteReports = useMemo(() => buildMinuteReports(samples), [samples]);
  const currentMinute = minuteReports.length > 0 ? minuteReports[minuteReports.length - 1] : null;

  const exportCsv = useCallback(() => {
    if (samples.length === 0) return;
    const blob = new Blob([samplesToCsv(samples)], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `milyonus-optic-tracking-report-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [samples]);

  const exportHighResLog = useCallback(async () => {
    const data = await recorder.dump();
    if (data.length === 0) return;
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `milyonus-optic-highres-log-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const start = useCallback(async () => {
    setPhase("loading");
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

      const createLandmarker = (delegate: "GPU" | "CPU") =>
        FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL, delegate },
          runningMode: "VIDEO",
          numFaces: MAX_FACES,
          minFaceDetectionConfidence: 0.3,
          minFacePresenceConfidence: 0.3,
          minTrackingConfidence: 0.3,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });
      const createDetector = (delegate: "GPU" | "CPU") =>
        FaceDetector.createFromOptions(vision, {
          baseOptions: { modelAssetPath: FACE_DETECTOR_MODEL, delegate },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.3,
        });

      let landmarker: FaceLandmarker;
      let detector: FaceDetector;
      try {
        [landmarker, detector] = await Promise.all([
          createLandmarker("GPU"),
          createDetector("GPU"),
        ]);
      } catch {
        [landmarker, detector] = await Promise.all([
          createLandmarker("CPU"),
          createDetector("CPU"),
        ]);
      }
      landmarkerRef.current = landmarker;
      detectorRef.current = detector;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      trackerRef.current.reset();
      lastFaceCountRef.current = 0;
      sessionStartRef.current = Date.now();
      lastGazeAwayLogRef.current = 0;
      setSamples([]);
      setPhase("live");
      setLog([
        {
          t: stamp(),
          emotion: "neutral",
          msg: "Camera link established — vision models online",
        },
      ]);
      rafRef.current = requestAnimationFrame(() => loop());
    } catch (err) {
      const e = err as Error;
      setErrorMsg(
        e.name === "NotAllowedError"
          ? "Camera access denied — enable camera permission in your browser and retry."
          : `Unable to start camera — ${e.message || "please try again."}`,
      );
      setPhase("error");
    }
  }, [loop]);

  useEffect(
    () => () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const v = videoRef.current;
      const s = v?.srcObject as MediaStream | null;
      s?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
      detectorRef.current?.close();
    },
    [],
  );

  const primary = faces.reduce<FaceReading | null>(
    (best, r) => (!best || r.area > best.area ? r : best),
    null,
  );
  const dominant = primary?.dominant ?? "neutral";
  const scores = primary?.scores ?? emptyScores();
  const accent = EMOTION_TOKEN[dominant];
  const liveGaze = gazeDirectionFromPose(primary?.pose ?? null);
  const liveFocusScore = focusScoreFromPose(primary?.pose ?? null);

  return (
    <div className="stage-light min-h-screen w-full">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 pb-16 pt-5 sm:px-8">
        {/* nav */}
        <header className="flex items-center justify-between gap-4">
          <MilyonusMark />
          <button
            onClick={() => setMuted((m) => !m)}
            className="btn-brand px-5 py-2 text-[11px] uppercase tracking-[0.18em]"
          >
            Sound {muted ? "Off" : "On"}
          </button>
        </header>

        {/* hero */}
        <div className="flex flex-col items-center gap-4 pt-4 text-center">
          <span className="pill-tag">Open Source · On-Device</span>
          <h1 className="max-w-3xl text-3xl font-semibold leading-tight tracking-tight text-foreground sm:text-5xl">
            Real-time optic tracking{" "}
            <em className="font-light italic text-[color:var(--silver)]">for the browser</em>
          </h1>
          <p className="max-w-xl text-xs leading-relaxed tracking-[0.12em] text-muted-foreground sm:text-sm">
            478-point face mesh, 3D head pose, multi-face tracking and focus scoring — read live in
            your browser. Nothing leaves this device.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1fr_310px]">
          {/* video stage */}
          <section className="relative">
            <div
              className="relative aspect-video w-full overflow-hidden rounded-3xl border bg-black"
              style={{ borderColor: `color-mix(in oklab, ${accent} 35%, transparent)` }}
            >
              <video
                ref={videoRef}
                playsInline
                muted
                className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
              />
              <canvas ref={canvasRef} className="absolute inset-0 h-full w-full object-cover" />

              <span className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 text-[10px] uppercase tracking-[0.35em] text-muted-foreground">
                Live · {faces.length} face{faces.length === 1 ? "" : "s"} · {fps} fps · {clock}
              </span>

              {/* callout */}
              {callout && (
                <div className="pointer-events-none absolute inset-x-4 top-8 flex justify-center">
                  <div className="animate-combo rounded-full bg-black/50 px-5 py-3 text-center text-base font-semibold leading-tight text-foreground backdrop-blur-md sm:text-2xl">
                    Milyonus detected{" "}
                    <em
                      className="font-light italic"
                      style={{
                        color: EMOTION_TOKEN[callout.e],
                        textShadow: `0 0 24px color-mix(in oklab, ${EMOTION_TOKEN[callout.e]} 55%, transparent)`,
                      }}
                    >
                      {callout.e}
                    </em>{" "}
                    here
                  </div>
                </div>
              )}

              {/* gate / loading / error overlays */}
              {phase !== "live" && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-5 bg-black/80 px-6 text-center backdrop-blur-sm">
                  {phase === "gate" && (
                    <>
                      <h2 className="text-xl font-semibold tracking-tight text-foreground sm:text-3xl">
                        Milyonus{" "}
                        <em className="font-light italic text-[color:var(--silver)]">
                          Optic Tracking 2.4
                        </em>
                      </h2>
                      <p className="max-w-md text-xs tracking-[0.12em] text-muted-foreground sm:text-sm">
                        Grant camera access to begin. All processing happens locally.
                      </p>
                      <button
                        onClick={() => void start()}
                        className="btn-brand px-8 py-3 text-sm tracking-[0.12em]"
                      >
                        Start
                      </button>
                    </>
                  )}
                  {phase === "loading" && (
                    <p className="animate-flicker text-xs uppercase tracking-[0.35em] text-muted-foreground">
                      Loading vision models…
                    </p>
                  )}
                  {phase === "error" && (
                    <>
                      <p className="max-w-md text-sm tracking-[0.06em] text-destructive">
                        {errorMsg}
                      </p>
                      <button
                        onClick={() => void start()}
                        className="btn-brand px-7 py-2.5 text-xs tracking-[0.14em]"
                      >
                        Retry
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </section>

          {/* right column */}
          <aside className="flex flex-col gap-5">
            <div className="glass-panel p-5">
              <h2 className="mb-4 text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
                Expression Matrix
              </h2>
              <div className="space-y-3">
                {EMOTIONS.map((e) => (
                  <div key={e}>
                    <div className="flex justify-between text-[11px] tracking-[0.14em] text-muted-foreground">
                      <span
                        className={e === dominant ? "text-foreground" : undefined}
                        style={{ color: e === dominant ? EMOTION_TOKEN[e] : undefined }}
                      >
                        {e}
                      </span>
                      <span>{(scores[e] * 100).toFixed(0)}%</span>
                    </div>
                    <div className="mt-1 h-[3px] w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full transition-[width] duration-150"
                        style={{
                          width: `${Math.min(100, scores[e] * 100)}%`,
                          background: EMOTION_TOKEN[e],
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-panel p-5">
              <h2 className="mb-3 text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
                Subject {primary ? primary.id : "--"} · 3D Pose
              </h2>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px] tracking-[0.1em]">
                {[
                  ["Faces", `${faces.length}`],
                  [
                    "Confidence",
                    primary?.confidence != null
                      ? `${(primary.confidence * 100).toFixed(0)}%`
                      : "--",
                  ],
                  ["Landmarks", primary ? `${primary.landmarkCount}` : "--"],
                  ["Yaw", primary ? `${primary.pose.yaw.toFixed(1)}°` : "--"],
                  ["Pitch", primary ? `${primary.pose.pitch.toFixed(1)}°` : "--"],
                  ["Roll", primary ? `${primary.pose.roll.toFixed(1)}°` : "--"],
                  ["Depth", primary ? primary.pose.distance.toFixed(2) : "--"],
                  ["Deep Tech", latestPrimaryRef.current.deepTechScore != null ? `${(latestPrimaryRef.current.deepTechScore * 100).toFixed(1)}%` : "--"],
                  ["EAR (L/R)", `${earL.toFixed(2)} / ${earR.toFixed(2)}`],
                  ["MAR", mar.toFixed(2)],
                  ["State", dominant],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-white/10 py-1">
                    <dt className="text-muted-foreground">{k}</dt>
                    <dd style={{ color: accent }}>{v}</dd>
                  </div>
                ))}
              </dl>
            </div>

            <div className="glass-panel p-5">
              <h2 className="mb-3 text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
                Focus Tracking
              </h2>
              <div className="flex items-center justify-between">
                <span
                  className="text-xs font-semibold tracking-[0.16em]"
                  style={{ color: cssVar(GAZE_COLOR[liveGaze]) }}
                >
                  {GAZE_LABEL[liveGaze]}
                </span>
                <span className="text-[11px] text-muted-foreground">
                  {(liveFocusScore * 100).toFixed(0)}%
                </span>
              </div>
              <div className="mt-1.5 h-[3px] w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full transition-[width] duration-150"
                  style={{
                    width: `${liveFocusScore * 100}%`,
                    background: cssVar(GAZE_COLOR[liveGaze]),
                  }}
                />
              </div>
              <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px] tracking-[0.1em]">
                {[
                  ["Samples/min", currentMinute ? `${currentMinute.sampleCount}` : "--"],
                  [
                    "Avg confidence",
                    currentMinute?.avgConfidence != null
                      ? `${(currentMinute.avgConfidence * 100).toFixed(0)}%`
                      : "--",
                  ],
                  ["Focused", currentMinute ? `${currentMinute.focusedPct.toFixed(0)}%` : "--"],
                  [
                    "Looking away",
                    currentMinute ? `${currentMinute.lookingAwayPct.toFixed(0)}%` : "--",
                  ],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-white/10 py-1">
                    <dt className="text-muted-foreground">{k}</dt>
                    <dd style={{ color: accent }}>{v}</dd>
                  </div>
                ))}
              </dl>
            </div>

            {faces.length > 0 && (
              <div className="glass-panel p-5">
                <h2 className="mb-3 text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
                  Tracked Faces
                </h2>
                <div className="space-y-2 text-[11px] tracking-[0.08em]">
                  {faces.map((f) => (
                    <div
                      key={f.id}
                      className="flex items-center justify-between rounded-lg border border-white/10 px-3 py-2"
                      style={{
                        borderColor:
                          f.id === primary?.id
                            ? `color-mix(in oklab, ${EMOTION_TOKEN[f.dominant]} 55%, transparent)`
                            : undefined,
                      }}
                    >
                      <span className="text-muted-foreground">ID {f.id}</span>
                      <span style={{ color: EMOTION_TOKEN[f.dominant] }}>{f.dominant}</span>
                      <span className="text-muted-foreground">
                        {f.confidence != null ? `${(f.confidence * 100).toFixed(0)}%` : "--"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </aside>
        </div>

        {/* log */}
        <section className="glass-panel p-5">
          <h2 className="mb-3 text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
            Detection Log
          </h2>
          <div className="h-40 overflow-y-auto pr-1 text-[11px] leading-relaxed tracking-[0.08em]">
            {log.length === 0 && <p className="text-muted-foreground">Awaiting signal…</p>}
            {log.map((l, i) => (
              <p key={`${l.t}-${i}`}>
                <span className="text-muted-foreground/60">{l.t}</span>{" "}
                <span style={{ color: EMOTION_TOKEN[l.emotion] }}>{l.msg}</span>
              </p>
            ))}
          </div>
        </section>

        {/* per-minute report */}
        <section className="glass-panel p-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
              Per-Minute Report
            </h2>
              <div className="flex gap-2">
                <button
                  onClick={exportCsv}
                  disabled={samples.length === 0}
                  className="btn-brand px-4 py-1.5 text-[10px] uppercase tracking-[0.16em] disabled:opacity-40"
                >
                  Export CSV
                </button>
                <button
                  onClick={exportHighResLog}
                  className="btn-brand px-4 py-1.5 text-[10px] uppercase tracking-[0.16em] opacity-80"
                >
                  High-Res Log
                </button>
              </div>
          </div>
          {minuteReports.length === 0 ? (
            <p className="text-[11px] text-muted-foreground">Awaiting first minute of samples…</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-left text-[11px] tracking-[0.06em]">
                <thead>
                  <tr className="text-muted-foreground">
                    <th className="border-b border-white/10 py-1.5 pr-3 font-normal">Minute</th>
                    <th className="border-b border-white/10 py-1.5 pr-3 font-normal">Samples</th>
                    <th className="border-b border-white/10 py-1.5 pr-3 font-normal">
                      Avg Confidence
                    </th>
                    <th className="border-b border-white/10 py-1.5 pr-3 font-normal">
                      Face Detected
                    </th>
                    <th className="border-b border-white/10 py-1.5 pr-3 font-normal">Focused</th>
                    <th className="border-b border-white/10 py-1.5 font-normal">Looking Away</th>
                  </tr>
                </thead>
                <tbody>
                  {[...minuteReports].reverse().map((m) => (
                    <tr key={m.minute}>
                      <td className="border-b border-white/5 py-1.5 pr-3">
                        {m.minute}
                        {m.minute === currentMinute?.minute && (
                          <span className="ml-1.5 text-[9px] text-muted-foreground">live</span>
                        )}
                      </td>
                      <td className="border-b border-white/5 py-1.5 pr-3">{m.sampleCount}</td>
                      <td className="border-b border-white/5 py-1.5 pr-3">
                        {m.avgConfidence != null ? `${(m.avgConfidence * 100).toFixed(0)}%` : "--"}
                      </td>
                      <td className="border-b border-white/5 py-1.5 pr-3">
                        {m.faceDetectedPct.toFixed(0)}%
                      </td>
                      <td
                        className="border-b border-white/5 py-1.5 pr-3"
                        style={{ color: "var(--happy)" }}
                      >
                        {m.focusedPct.toFixed(0)}%
                      </td>
                      <td
                        className="border-b border-white/5 py-1.5"
                        style={{ color: "var(--angry)" }}
                      >
                        {m.lookingAwayPct.toFixed(0)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
