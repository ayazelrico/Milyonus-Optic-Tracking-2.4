# Milyonus Optic Tracking 2.4

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Real-time, fully on-device facial optic tracking for the browser: a 478-point
face mesh, 3D head pose, per-face confidence scoring, multi-face tracking
with persistent IDs, and a focus-tracking system that samples attention once
per second and rolls it up into a per-minute report.

Nothing leaves the device — every frame is processed locally with
WebAssembly/WebGL, in-browser, and no image or video data is ever uploaded
anywhere.

## Table of contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Focus Tracking System](#focus-tracking-system)
- [CSV export format](#csv-export-format)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Privacy & models](#privacy--models)
- [Tech stack](#tech-stack)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- **478-point face mesh** — full tesselation, contours and iris landmarks
  (468 face points + 10 iris points), rendered live over the mirrored camera
  feed with a per-emotion accent color.
- **3D head pose** — yaw / pitch / roll decomposed from the model's raw
  rotation matrix, plus a relative depth estimate from its translation,
  visualized as a live RGB axis gizmo at the nose tip.
- **Confidence score** — a genuine per-face detection-confidence value read
  from a dedicated, independent face detector and matched to each tracked
  face by nearest-centroid — not a synthetic or guessed number.
- **Face count & multi-face support** — detects and tracks up to 4 faces per
  frame simultaneously, each with its own mesh, pose, confidence and ID.
- **Track ID** — a lightweight, from-scratch greedy nearest-centroid tracker
  keeps a stable numeric ID per face across frames. The underlying model has
  no built-in persistent ID for multi-face scenes, so this project adds one.
- **Focus Tracking Score** — a continuous 0–1 score derived from head yaw and
  pitch, plus a discrete gaze label (`center` / `left` / `right` / `up` /
  `down` / `none`). A throttled "looking away" alert is written to the log
  whenever the subject turns away from the screen.
- **Per-second recording** — confidence and focus/gaze are sampled on a
  clean, independent 1Hz timer (not tied to the render loop's frame rate),
  so the resulting time series is exactly one sample per second.
- **Per-minute report** — every second-level sample is aggregated live into
  a per-minute table: sample count, average confidence, % of seconds a face
  was detected, % focused, % looking away.
- **CSV export** — one click downloads the entire per-second time series as
  a CSV file for offline analysis, spreadsheets, or a BI pipeline.
- **Expression matrix** — an 8-bucket "vibe" reading (smile, sad, angry,
  surprised, fearful, disgusted, thinking, neutral) derived from 52
  ARKit-style blendshape coefficients, with arcade-style combo callouts and
  optional sound effects.
- **GPU-accelerated with automatic CPU fallback** — both vision models try
  to initialize on the GPU delegate first and transparently fall back to CPU
  if WebGL2 isn't available.

## Screenshots

_Add your own screenshots or a short GIF here once you've run the app
locally — a capture of the live mesh + HUD, the Focus Tracking panel, and
the per-minute report table are good candidates. None are checked in yet so
the README doesn't ship placeholder or third-party imagery._

## Quick start

Requires Node.js 18+ (or [Bun](https://bun.sh)).

```sh
git clone <this-repository-url>
cd milyonus-optic-tracking
npm install
npm run dev
```

Open the printed local URL, click **Start**, and grant camera access. Models
load from the network on first run (see [Privacy & models](#privacy--models)
below) and are cached by the browser after that.

Other scripts:

```sh
npm run build     # production build (outputs to dist/)
npm run preview   # preview the production build locally
npm run lint       # eslint
npm run format     # prettier --write
```

## How it works

The app runs two [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision)
graphs on every video frame, entirely in-browser, in `VIDEO` running mode:

- **`FaceLandmarker`** — up to 4 faces per frame, each with 478 normalized
  landmarks, 52 blendshape coefficients (`outputFaceBlendshapes: true`), and
  a 4×4 facial transformation matrix (`outputFacialTransformationMatrixes:
  true`).
- **`FaceDetector`** — a lighter, separate model that returns axis-aligned
  bounding boxes with a genuine `categories[0].score` detection confidence.
  `FaceLandmarker` alone doesn't expose a confidence score, so this project
  runs the detector in parallel purely to source a real one, then pairs each
  detection to the nearest landmark-derived face by bounding-box centroid.

On top of the raw model output, this project adds:

- **`src/lib/facePose.ts`** — decomposes the column-major 4×4 transformation
  matrix into yaw / pitch / roll using a "YXZ" Euler decomposition (the same
  math as `THREE.Matrix4.setFromRotationMatrix`, which is what the
  column-major layout is designed for), plus a depth proxy read straight
  from the matrix's Z translation.
- **`src/lib/faceTracker.ts`** — a minimal greedy nearest-centroid tracker.
  Every frame, each detected face's box is matched to the closest existing
  track (within a distance threshold relative to face size); unmatched
  tracks are dropped after ~600ms unseen, and unmatched faces get a new ID.
  This is what makes a face's ID stable across frames even though the model
  itself re-detects from scratch every frame.
- **`src/lib/blendshapeEmotion.ts`** — heuristically combines the 52
  blendshape coefficients (`mouthSmileLeft`, `browDownLeft`, `jawOpen`,
  `noseSneerLeft`, etc.) into 8 expression scores, weighting each bucket
  by the blendshapes that most plausibly compose that expression, then
  derives a `neutral` remainder score.
- **`src/lib/focusTracking.ts`** — turns head pose into a gaze label and a
  0–1 focus score (see [Focus Tracking System](#focus-tracking-system)
  below), records one sample per second, and aggregates the running history
  into per-minute reports.

All of the drawing (mesh, contours, irises, the pose gizmo, per-face target
brackets and ID/confidence labels) happens on a single `<canvas>` layered
over the mirrored `<video>` element. The canvas content is drawn unmirrored
and then flipped with a single `ctx.scale(-1, 1)` transform so it lines up
visually with the mirrored video, while text labels are drawn afterwards,
unflipped, so they stay readable instead of appearing backwards.

## Focus Tracking System

The focus system answers one question every second: **is the subject
looking at the screen?**

1. **Gaze classification** (`gazeDirectionFromPose`) — checks head yaw and
   pitch against fixed thresholds:
   - `|yaw| > 15°` → `left` or `right` (sign-dependent)
   - else `|pitch| > 12°` → `up` or `down`
   - else → `center`
   - no face detected → `none`
2. **Focus score** (`focusScoreFromPose`) — a continuous 0–1 value, not just
   the discrete label above: it linearly penalizes yaw beyond 15° and pitch
   beyond 12°, bottoming out at `0` once either axis passes 35° (yaw) or 30°
   (pitch), and is `0` outright when no face is present.
3. **Recording** — a `setInterval` timer independent of the render loop
   samples the *current* primary subject's confidence, yaw, pitch, gaze
   label and focus score exactly once per second, tagging each sample with
   an elapsed-seconds counter and an ISO timestamp.
4. **Alerts** — whenever the gaze flips to `left` or `right`, a log line is
   written (throttled to at most once every 4 seconds) with the current yaw
   angle and confidence, so a reviewer scanning the log can see every
   "looked away" moment without being flooded by duplicates.
5. **Per-minute rollup** (`buildMinuteReports`) — groups all recorded
   samples by `floor(second / 60)` and computes, per minute: sample count,
   average confidence (over seconds where a face was present), % of seconds
   a face was detected at all, average focus score, % of seconds classified
   `center` ("focused"), and % classified `left`/`right` ("looking away").
   The table in the UI re-renders live as new samples land, with the
   in-progress minute marked `live`.

These thresholds are deliberately simple, fixed constants rather than a
trained classifier — tune them in `src/lib/focusTracking.ts` for your own
camera angle, seating distance, or strictness requirements.

## CSV export format

The **Export CSV** button downloads every recorded second as a row:

| Column        | Meaning                                             |
| -------------- | ---------------------------------------------------- |
| `second`       | Seconds elapsed since the session started            |
| `timestamp`    | ISO 8601 timestamp of the sample                      |
| `has_face`     | `1` if a face was present that second, else `0`       |
| `confidence`   | Detection confidence (0–1), blank if no face          |
| `yaw_deg`      | Head yaw in degrees, blank if no face                 |
| `pitch_deg`    | Head pitch in degrees, blank if no face                |
| `gaze`         | `center` / `left` / `right` / `up` / `down` / `none`  |
| `focus_score`  | 0–1 continuous focus score for that second             |

## Project structure

```
├── index.html                    # Vite entry point, page metadata
├── src/
│   ├── main.tsx                  # React root
│   ├── App.tsx                   # Thin wrapper around the HUD
│   ├── styles.css                # Tailwind v4 theme tokens + HUD utilities
│   ├── components/
│   │   ├── FacecamHUD.tsx        # The whole app: capture, models, render loop, UI
│   │   └── MilyonusMark.tsx      # Self-contained text wordmark (no image asset)
│   └── lib/
│       ├── facePose.ts           # Rotation matrix → yaw/pitch/roll/depth
│       ├── faceTracker.ts        # Nearest-centroid multi-face tracker
│       ├── blendshapeEmotion.ts  # Blendshapes → 8-bucket expression scores
│       ├── focusTracking.ts      # Gaze/focus scoring, sampling, per-minute reports
│       └── utils.ts              # `cn()` className helper
└── public/                       # Static assets served as-is (favicon, robots.txt)
```

## Configuration

Everything tunable lives as named constants near the top of the relevant
file — there's no separate config file or environment variables to manage:

| What                                              | Where                                      |
| --------------------------------------------------- | --------------------------------------------- |
| Max simultaneous tracked faces (default `4`)        | `MAX_FACES` in `FacecamHUD.tsx`               |
| Model confidence thresholds                          | `minFaceDetectionConfidence` / `minFacePresenceConfidence` / `minTrackingConfidence` / `minDetectionConfidence` in `FacecamHUD.tsx`'s `start()` |
| Gaze / focus-score thresholds                        | `YAW_AWAY_DEG`, `PITCH_AWAY_DEG`, `YAW_FULL_OFF_DEG`, `PITCH_FULL_OFF_DEG` in `focusTracking.ts` |
| Track loss timeout / matching distance                | constructor args of `FaceTracker` in `faceTracker.ts` |
| Expression callout threshold (`72%` confidence)        | the `conf > 0.72` check in `FacecamHUD.tsx`'s render loop |
| Model asset URLs / WASM runtime version              | `FACE_LANDMARKER_MODEL`, `FACE_DETECTOR_MODEL`, `TASKS_VISION_VERSION` in `FacecamHUD.tsx` |

## Requirements

- A browser with camera access and WebAssembly support (any current Chrome,
  Edge, Safari or Firefox).
- WebGL2 for the GPU delegate — the app automatically falls back to CPU
  inference if it isn't available, at a lower frame rate.
- A camera. That's it — no server, no database, no account.

## Privacy & models

All inference runs locally in the browser via WebAssembly/WebGL — no video
frame or image is ever sent to a server, and there is no backend in this
project at all. The only network requests this app makes are to fetch the
MediaPipe WASM runtime (from `jsdelivr.net`) and the two model files (from
`storage.googleapis.com`) on first load; after that, everything runs
offline from the browser cache. Recorded per-second samples and the CSV
export live only in the page's memory for the current session and are never
transmitted anywhere — closing the tab discards them unless you export.

## Tech stack

React 19, TypeScript, Vite 8, Tailwind CSS v4, and
[`@mediapipe/tasks-vision`](https://www.npmjs.com/package/@mediapipe/tasks-vision).
No backend, no router, no state-management library, no UI component
library — the whole app is one focused component plus a handful of small,
independently testable `src/lib` modules.

## Roadmap

Ideas that fit the project but aren't implemented yet — contributions
welcome:

- Session recording/replay (scrub back through a past run's samples).
- Configurable alert rules (e.g. "log an event only after N consecutive
  seconds looking away").
- A settings panel for the currently hardcoded thresholds in
  [Configuration](#configuration).
- WebRTC/screen-share integration for real video-call use cases.
- Unit tests for `facePose`, `faceTracker`, `blendshapeEmotion` and
  `focusTracking` (all pure functions, straightforward to test).

## Contributing

Issues and pull requests are welcome. Before opening a PR:

```sh
npm run lint
npm run build
```

Keep changes scoped and avoid adding new dependencies unless they're clearly
worth their weight — this project intentionally has a small, auditable
dependency footprint.

## FAQ

**Does this send my camera feed anywhere?**
No. Everything runs in your browser via WebAssembly/WebGL. See
[Privacy & models](#privacy--models).

**Why does confidence come from a separate `FaceDetector` instead of
`FaceLandmarker`?**
`FaceLandmarker`'s result type doesn't include a per-detection confidence
score. Running the lighter `FaceDetector` in parallel is the only way to
attach a real, model-reported confidence value rather than a guessed one.

**Why doesn't this include age/gender estimation?**
MediaPipe Face Landmarker doesn't provide it. Earlier prototypes of this
project used a different library that did, but that was dropped along with
that dependency — everything documented here reflects what's actually in
`src/`.

**The face count says 0 even though I'm on camera.**
Check the browser's camera permission for the site, and confirm your camera
isn't in use by another app/tab. WebGL2 support is also required for the
default GPU path (see [Requirements](#requirements)).

## Acknowledgements

Built on [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
and [MediaPipe Face Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)
(Google AI Edge), used via the public `@mediapipe/tasks-vision` package and
its hosted model files, under their own license.

## License

[MIT](./LICENSE) — © Milyonus.
