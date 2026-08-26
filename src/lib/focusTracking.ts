import type { HeadPose } from "./facePose";

export type GazeDirection = "center" | "left" | "right" | "up" | "down" | "none";

/** Beyond this yaw the candidate is considered turned away from the screen. */
const YAW_AWAY_DEG = 15;
/** Beyond this pitch the candidate is considered looking away (down at a phone, etc.). */
const PITCH_AWAY_DEG = 12;
/** Yaw/pitch magnitude at which the focus score bottoms out at 0. */
const YAW_FULL_OFF_DEG = 35;
const PITCH_FULL_OFF_DEG = 30;

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** Classifies where the candidate is looking, relative to the screen. */
export function gazeDirectionFromPose(pose: HeadPose | null): GazeDirection {
  if (!pose) return "none";
  if (Math.abs(pose.yaw) > YAW_AWAY_DEG) return pose.yaw > 0 ? "right" : "left";
  if (Math.abs(pose.pitch) > PITCH_AWAY_DEG) return pose.pitch > 0 ? "down" : "up";
  return "center";
}

/** 0 (fully turned away / no face) .. 1 (squarely facing the camera). */
export function focusScoreFromPose(pose: HeadPose | null): number {
  if (!pose) return 0;
  const yawPenalty = clamp01(Math.abs(pose.yaw) / YAW_FULL_OFF_DEG);
  const pitchPenalty = clamp01(Math.abs(pose.pitch) / PITCH_FULL_OFF_DEG);
  return clamp01(1 - Math.max(yawPenalty, pitchPenalty));
}

export interface SecondSample {
  /** Seconds elapsed since the session started. */
  tSec: number;
  atISO: string;
  hasFace: boolean;
  confidence: number | null;
  yaw: number | null;
  pitch: number | null;
  distance: number | null;
  area: number | null;
  deepTechScore: number | null;
  gaze: GazeDirection;
  focusScore: number;
}

export interface MinuteReport {
  minute: number;
  sampleCount: number;
  avgConfidence: number | null;
  faceDetectedPct: number;
  avgFocusScore: number;
  focusedPct: number;
  lookingAwayPct: number;
}

/** Groups per-second samples into per-minute aggregates, sorted ascending. */
export function buildMinuteReports(samples: readonly SecondSample[]): MinuteReport[] {
  const byMinute = new Map<number, SecondSample[]>();
  for (const s of samples) {
    const minute = Math.floor(s.tSec / 60);
    const list = byMinute.get(minute);
    if (list) list.push(s);
    else byMinute.set(minute, [s]);
  }

  return Array.from(byMinute.entries())
    .sort(([a], [b]) => a - b)
    .map(([minute, list]) => {
      const withFace = list.filter((s) => s.hasFace);
      const confidences = withFace.map((s) => s.confidence).filter((c): c is number => c != null);
      const focused = list.filter((s) => s.gaze === "center").length;
      const away = list.filter((s) => s.gaze === "left" || s.gaze === "right").length;
      const avgFocusScore = list.reduce((sum, s) => sum + s.focusScore, 0) / list.length;

      return {
        minute,
        sampleCount: list.length,
        avgConfidence: confidences.length
          ? confidences.reduce((a, b) => a + b, 0) / confidences.length
          : null,
        faceDetectedPct: (withFace.length / list.length) * 100,
        avgFocusScore,
        focusedPct: (focused / list.length) * 100,
        lookingAwayPct: (away / list.length) * 100,
      };
    });
}

export function samplesToCsv(samples: readonly SecondSample[]): string {
  const header = "second,timestamp,has_face,confidence,yaw_deg,pitch_deg,distance,area,deep_tech_score,gaze,focus_score\n";
  const rows = samples.map((s) =>
    [
      s.tSec,
      s.atISO,
      s.hasFace ? 1 : 0,
      s.confidence != null ? s.confidence.toFixed(4) : "",
      s.yaw != null ? s.yaw.toFixed(2) : "",
      s.pitch != null ? s.pitch.toFixed(2) : "",
      s.distance != null ? s.distance.toFixed(2) : "",
      s.area != null ? s.area.toFixed(0) : "",
      s.deepTechScore != null ? s.deepTechScore.toFixed(4) : "",
      s.gaze,
      s.focusScore.toFixed(4),
    ].join(","),
  );
  return header + rows.join("\n");
}
