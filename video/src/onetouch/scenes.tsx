import React from "react";
import { AbsoluteFill, Easing, Img, interpolate, Sequence, staticFile, useCurrentFrame } from "remotion";
import { Video } from "@remotion/media";
import { C, FONT, Kicker, MONO, Reveal, rise, useSceneFade, vis } from "./ui";

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

/** Black backdrop; `arc` adds the faint trajectory arc (only in the later scenes). */
const Backdrop: React.FC<{ progress?: number; arc?: boolean }> = ({ progress = 0.5, arc = false }) => {
  const W = 1920;
  const H = 1080;
  const cx = W / 2;
  const cy = H + 1500;
  const r = 1720;
  const a0 = Math.PI * 1.32;
  const a1 = Math.PI * 1.68;
  const a = a0 + (a1 - a0) * progress;
  const pt = (t: number) => [cx + r * Math.cos(t), cy + r * Math.sin(t)];
  const [x0, y0] = pt(a0);
  const [x1, y1] = pt(a1);
  const [xd, yd] = pt(a);
  return (
    <AbsoluteFill style={{ background: "#000" }}>
      {arc && (
        <svg width={W} height={H} style={{ position: "absolute", inset: 0 }}>
          <path d={`M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`} stroke="rgba(255,255,255,0.16)" strokeWidth={2} fill="none" />
          <path d={`M ${x0} ${y0} A ${r} ${r} 0 0 1 ${xd} ${yd}`} stroke="rgba(255,255,255,0.6)" strokeWidth={2.5} fill="none" />
          <circle cx={xd} cy={yd} r={7} fill="#fff" />
        </svg>
      )}
    </AbsoluteFill>
  );
};

/** An app screen recording (1920x1180 clip) filling the frame: the top 100 px are cropped, so
 * the app's bottom HUD (the stats) stays fully visible. */
const Screen: React.FC<{ src: string; style?: React.CSSProperties }> = ({ src, style }) => (
  <div style={{ position: "absolute", left: 0, top: -100, width: 1920, height: 1180, ...style }}>
    <Video src={staticFile(src)} objectFit="fill" muted style={{ width: "100%", height: "100%" }} />
  </div>
);

/** Small caption at the top-left (the recordings' empty top area, never over the stats). */
const TopCaption: React.FC<{ label?: string; text: string; opacity: number }> = ({ label, text, opacity }) => (
  <div style={{ position: "absolute", left: 80, top: 30, opacity }}>
    {label && (
      <div style={{ fontFamily: MONO, fontSize: 24, letterSpacing: "0.24em", textTransform: "uppercase", color: C.go }}>
        {label}
      </div>
    )}
    <div
      style={{
        fontFamily: FONT,
        fontWeight: 500,
        fontSize: 48,
        color: C.text,
        textShadow: "0 2px 12px rgba(0,0,0,0.9)",
        marginTop: label ? 4 : 0,
      }}
    >
      {text}
    </div>
  </div>
);

/** One-line caption in the gap between the scale grid's tiles and its stats HUD. */
const GapCaption: React.FC<{ tag?: string; text: string; opacity: number }> = ({ tag, text, opacity }) => (
  <div style={{ position: "absolute", left: 0, right: 0, top: 808, textAlign: "center", opacity }}>
    <span style={{ fontFamily: FONT, fontWeight: 500, fontSize: 38, color: C.text }}>
      {tag && <span style={{ fontFamily: MONO, fontSize: 24, letterSpacing: "0.2em", textTransform: "uppercase", color: C.go, marginRight: 22 }}>{tag}</span>}
      {text}
    </span>
  </div>
);

const Credit: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div
    style={{
      position: "absolute",
      right: 60,
      bottom: 40,
      fontFamily: MONO,
      fontSize: 18,
      color: "rgba(255,255,255,0.7)",
      textShadow: "0 1px 6px rgba(0,0,0,0.9)",
    }}
  >
    {children}
  </div>
);

/* ================================================================== OPENING */

/** Both of today's methods side by side, then the verdict, big. */
const Clip: React.FC<{ label: string; children: React.ReactNode; at: number }> = ({ label, children, at }) => {
  const frame = useCurrentFrame();
  return (
    <div style={{ opacity: rise(frame, at, at + 16) }}>
      <div style={{ position: "relative", width: 860, height: 484, borderRadius: 8, overflow: "hidden", background: "#111" }}>
        {children}
      </div>
      <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 38, color: C.text, marginTop: 22 }}>{label}</div>
    </div>
  );
};

const VERDICT = ["Slow.", "Expensive.", "Inaccessible."];
const V0 = 135; // first verdict word (frame)
const VSTEP = 22;

export const Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade(14, 14);
  const dim = interpolate(frame, [V0 - 14, V0], [0, 0.82], clamp);
  const pop = (at: number) => ({
    opacity: interpolate(frame, [at, at + 3], [0, 1], clamp),
    scale: String(interpolate(frame, [at, at + 8], [1.18, 1], { ...clamp, easing: Easing.out(Easing.cubic) })),
  });
  const last = V0 + VERDICT.length * VSTEP + 12;
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000" }}>
      <div style={{ position: "absolute", left: 0, right: 0, top: 120, textAlign: "center", opacity: rise(frame, 4, 20) }}>
        <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 60, color: C.text }}>How robots learn today</div>
      </div>
      <div style={{ position: "absolute", left: 80, right: 80, top: 250, display: "flex", justifyContent: "space-between" }}>
        <Clip label="Teleoperation rigs" at={14}>
          <Video src={staticFile("clips/teleop.mp4")} loop objectFit="cover" muted style={{ width: "100%", height: "100%" }} />
        </Clip>
        <Clip label="Motion-capture gloves" at={30}>
          {/* scaled up so the clip's burned-in title (bottom) is cropped */}
          <div style={{ position: "absolute", left: -119, top: -30, width: 1118, height: 600 }}>
            <Video src={staticFile("clips/gloves.mp4")} loop objectFit="fill" muted style={{ width: "100%", height: "100%" }} />
          </div>
        </Clip>
      </div>
      <Credit>Gloves: Dexta Robotics (Dexmo)</Credit>

      {/* the verdict */}
      <AbsoluteFill style={{ background: `rgba(0,0,0,${dim})`, justifyContent: "center", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 40 }}>
          {VERDICT.map((w, i) => (
            <div key={w} style={{ fontFamily: FONT, fontWeight: 700, fontSize: 124, color: C.text, ...pop(V0 + i * VSTEP) }}>
              {w}
            </div>
          ))}
        </div>
        <div
          style={{
            marginTop: 40,
            fontFamily: FONT,
            fontWeight: 700,
            fontSize: 72,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "#000",
            background: C.warn,
            padding: "10px 36px",
            ...pop(last),
          }}
        >
          Big labs only
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

export const DataGap: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade();
  const grow = interpolate(frame, [30, 78], [0, 1], { ...clamp, easing: Easing.bezier(0.5, 0, 0.2, 1) });
  const row = (label: string, value: React.ReactNode, bar: React.ReactNode, at: number) => (
    <div style={{ marginTop: 60, opacity: rise(frame, at, at + 14) }}>
      <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 52, color: C.dim }}>{label}</div>
      <div style={{ position: "relative", height: 46, marginTop: 14 }}>{bar}</div>
      <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 64, color: C.text, marginTop: 12 }}>{value}</div>
    </div>
  );
  return (
    <AbsoluteFill style={{ opacity: o }}>
      <Backdrop />
      <AbsoluteFill style={{ padding: "140px 160px 0" }}>
        <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 80, color: C.text, opacity: rise(frame, 4, 18) }}>
          Robots are starving for data.
        </div>
        {row(
          "A top robot model",
          <>
            10,000 hours
            <span style={{ fontWeight: 600, fontSize: 44, color: C.warn, marginLeft: 36, opacity: rise(frame, 120, 130) }}>
              75,000× less data
            </span>
          </>,
          <div style={{ position: "absolute", left: 0, top: 0, height: 46, width: 8, background: C.text }} />,
          10,
        )}
        {row(
          "A large language model",
          <>≈ {fmt(750_000_000 * grow)} hours*</>,
          <div style={{ position: "absolute", left: 0, top: 0, height: 46, width: 8 + grow * 2400, background: "rgba(255,255,255,0.85)" }} />,
          22,
        )}
      </AbsoluteFill>
      {/* narration */}
      <div style={{ position: "absolute", left: 160, right: 160, top: 850, fontFamily: FONT, fontWeight: 500, fontSize: 50, color: C.text }}>
        <div style={{ position: "absolute", opacity: vis(frame, 30, 118, 8) }}>LLMs got smart by learning from the whole internet.</div>
        <div style={{ position: "absolute", opacity: rise(frame, 120, 130) }}>
          Robots have no internet. <span style={{ color: C.warn }}>Every demo needs a lab.</span>
        </div>
      </div>
      <div style={{ position: "absolute", left: 160, bottom: 56, fontFamily: MONO, fontSize: 18, color: C.faint, opacity: rise(frame, 46, 62) }}>
        Robot model: π0, Physical Intelligence, 2024. *Our estimate: Llama 3's 15T training tokens as reading time.
      </div>
    </AbsoluteFill>
  );
};

export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade(14, 14);
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000", justifyContent: "center", alignItems: "center", textAlign: "center" }}>
      <Img
        src={staticFile("logo.png")}
        style={{ height: 300, opacity: rise(frame, 2, 20), scale: String(0.9 + 0.1 * rise(frame, 2, 26)) }}
      />
      <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 110, letterSpacing: "0.3em", marginRight: "-0.3em", color: C.text, marginTop: 30, opacity: rise(frame, 12, 30) }}>
        ONEGRASP
      </div>
      <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 64, color: C.text, marginTop: 30, opacity: rise(frame, 34, 52) }}>
        Teach robots by just doing the task.
      </div>
      <div style={{ fontFamily: MONO, fontSize: 28, letterSpacing: "0.3em", textTransform: "uppercase", color: C.go, marginTop: 34, opacity: rise(frame, 80, 98) }}>
        One webcam · No sensors · Few minutes
      </div>
    </AbsoluteFill>
  );
};

/* ================================================================== THE DEMO (one continuous run) */
export const LiveCard: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade(10, 10);
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000", justifyContent: "center", alignItems: "center", textAlign: "center" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 22, opacity: rise(frame, 2, 16) }}>
        <div style={{ width: 26, height: 26, borderRadius: 13, background: C.warn, opacity: 0.5 + 0.5 * Math.cos(frame / 6) }} />
        <div style={{ fontFamily: FONT, fontWeight: 700, fontSize: 110, letterSpacing: "0.12em", color: C.text }}>LIVE DEMO</div>
      </div>
      <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 52, color: C.text, marginTop: 28, opacity: rise(frame, 14, 28) }}>
        One uncut, continuous take.
      </div>
      <div style={{ fontFamily: MONO, fontSize: 24, letterSpacing: "0.24em", textTransform: "uppercase", color: C.dim, marginTop: 26, opacity: rise(frame, 30, 44) }}>
        Only the waiting is sped up
      </div>
    </AbsoluteFill>
  );
};


// clip lengths at 30 fps
const LIVE = 394; // 3.5-16.6 s of the take, real time
const PROC = 245; // 16.6-81.0 s, 8x (the wait)
const TRANSFER = 517; // 80.4-97.6 s, real time: GO FOR TRANSFER + the side-by-side playback
const SCALE_A = 148; // best_scale 0-14 s, 3.5x (empty tiles, the wait)
const SCALE_B = 290; // best_scale 14-23.7 s, real time: the tiles finish, 12/12; ends zooming into tile 02
const TILE = 529; // tile 05 (can lying down), its full replay at 1.5x, cropped to full screen (zoominforscaale.mov)
const ZOOM = 26; // frames of the zoom from the grid into tile 02
export const DEMO_FRAMES = LIVE + PROC + TRANSFER + SCALE_A + SCALE_B + TILE;

/** Large narration in the middle of the frame (over a dimmed recording). */
const BigLine: React.FC<{ label: string; children: React.ReactNode; opacity: number; color?: string }> = ({
  label,
  children,
  opacity,
  color = C.go,
}) => (
  <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", textAlign: "center", opacity, padding: "0 120px" }}>
    <AbsoluteFill style={{ background: "radial-gradient(ellipse 62% 26% at 50% 50%, rgba(0,0,0,0.7), rgba(0,0,0,0))" }} />
    <div style={{ position: "relative", fontFamily: MONO, fontSize: 36, letterSpacing: "0.26em", textTransform: "uppercase", color }}>{label}</div>
    <div style={{ position: "relative", fontFamily: FONT, fontWeight: 600, fontSize: 94, lineHeight: 1.12, color: C.text, marginTop: 20, textShadow: "0 2px 6px rgba(0,0,0,0.9), 0 4px 30px rgba(0,0,0,0.9)" }}>
      {children}
    </div>
  </AbsoluteFill>
);

/** One line of narration on a dark band at the bottom of the frame. */
const Narration: React.FC<{ label?: string; children: React.ReactNode; opacity: number }> = ({ label, children, opacity }) => (
  <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 230, opacity }}>
    <AbsoluteFill style={{ background: "linear-gradient(rgba(0,0,0,0), #000 38%)" }} />
    <div style={{ position: "absolute", left: 0, right: 0, bottom: 56, textAlign: "center" }}>
      {label && (
        <div style={{ fontFamily: MONO, fontSize: 24, letterSpacing: "0.24em", textTransform: "uppercase", color: C.go, marginBottom: 10 }}>
          {label}
        </div>
      )}
      <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 52, color: C.text }}>{children}</div>
    </div>
  </div>
);

export const Demo: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade(10, 14);
  const t1 = LIVE;
  const t2 = t1 + PROC;
  const t3 = t2 + TRANSFER;
  const t4 = t3 + SCALE_A;
  const t5 = t4 + SCALE_B;
  // zoom the grid into tile 05 (x 44-489, y 289-539 of the frame) over the last ZOOM frames
  const z = interpolate(frame, [t5 - ZOOM, t5], [0, 1], { ...clamp, easing: Easing.inOut(Easing.cubic) });
  const k = 1 + (1920 / 445 - 1) * z;
  const q = Math.round(TILE / 4); // one narration line per quarter of the tile run
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000" }}>
      <Sequence durationInFrames={LIVE}>
        <Screen src="clips/live.mp4" />
      </Sequence>
      <Sequence from={t1} durationInFrames={PROC}>
        <Screen src="clips/proc8.mp4" />
      </Sequence>
      <Sequence from={t2} durationInFrames={TRANSFER}>
        <Screen src="clips/transfer.mp4" />
      </Sequence>
      <Sequence from={t3} durationInFrames={SCALE_A}>
        <Screen src="clips/scale_a.mp4" />
      </Sequence>
      <Sequence from={t4} durationInFrames={SCALE_B}>
        <AbsoluteFill style={{ transformOrigin: "0 0", transform: `translate(${-44 * k * z}px, ${-289 * k * z}px) scale(${k})` }}>
          <Screen src="clips/scale_b.mp4" />
        </AbsoluteFill>
      </Sequence>
      <Sequence from={t5} durationInFrames={TILE}>
        <Video src={staticFile("clips/tile05.mp4")} objectFit="fill" muted style={{ width: "100%", height: "100%" }} />
        {/* fixed band over the tile's own bottom label, so it never shows between narration lines */}
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 230, background: "linear-gradient(rgba(0,0,0,0), #000 38%)" }} />
      </Sequence>

      {/* real-world view of the setup (phone camera, a separate run), bottom right */}
      <Sequence from={30} durationInFrames={LIVE - 30}>
        <div
          style={{
            position: "absolute",
            right: 40,
            bottom: 40,
            width: 576,
            height: 324,
            borderRadius: 10,
            overflow: "hidden",
            border: "2px solid rgba(255,255,255,0.7)",
            boxShadow: "0 10px 40px rgba(0,0,0,0.6)",
            opacity: vis(frame, 30, LIVE, 12),
          }}
        >
          <Video src={staticFile("clips/real_pip.mp4")} objectFit="cover" muted style={{ width: "100%", height: "100%" }} />
          <div style={{ position: "absolute", left: 14, top: 10, fontFamily: MONO, fontSize: 18, letterSpacing: "0.2em", textTransform: "uppercase", color: "#fff", textShadow: "0 1px 6px rgba(0,0,0,0.9)" }}>
            The setup · real-world view
          </div>
        </div>
      </Sequence>
      {/* live take */}
      <TopCaption label="Step 1 · Record" text="Do the task normally, with your own hand" opacity={vis(frame, 10, 185, 8)} />
      <TopCaption label="Step 1 · Record" text="Task: pour from the can, then set it down" opacity={vis(frame, 190, 362, 8)} />
      {/* the waits: a speed tag */}
      <TopCaption label="Step 2 · ▶▶ 8×" text="Hand → robot motion" opacity={vis(frame, 364, t2, 6)} />

      {/* transfer: the highlight, real time, no zoom */}
      <div style={{ position: "absolute", left: 0, right: 0, top: 690, textAlign: "center", opacity: vis(frame, t2 + 90, t3, 10) }}>
        <div style={{ fontFamily: MONO, fontSize: 26, letterSpacing: "0.24em", textTransform: "uppercase", color: C.go }}>Step 3 · Go for transfer</div>
        <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 56, color: C.text, marginTop: 8 }}>My hand → the robot, in sync</div>
      </div>
      {/* scale results */}
      {/* go for scale: big narration over the dimmed grid, then the grid at full brightness for 12/12 */}
      <AbsoluteFill style={{ background: "#000", opacity: 0.45 * Math.min(vis(frame, t3, t5 - ZOOM - 40, 12), 1) }} />
      <BigLine label="Step 4 · Why scale?" opacity={vis(frame, t3 + 4, t3 + 112, 10)}>
        One demo only teaches the robot one exact scene.
      </BigLine>
      <BigLine label="Why scale?" opacity={vis(frame, t3 + 112, t3 + 220, 10)}>
        Move the can or tip it over, and the robot is lost.
      </BigLine>
      <BigLine label="The impact" opacity={vis(frame, t3 + 220, t5 - ZOOM - 40, 10)}>
        OneGrasp turns 1 human demo into 12 robot demos, with no extra human time.
      </BigLine>
      <GapCaption text="12 / 12 succeeded" opacity={vis(frame, t5 - ZOOM - 40, t5 - ZOOM, 8)} />
      <TopCaption label="▶▶ 3.5×" text="" opacity={vis(frame, t3, t4, 6)} />
      {/* scene 02, zoomed: why scaling */}
      <Narration label="How it augments · ▶▶ 1.5×" opacity={vis(frame, t5, t5 + q, 10)}>
        Half start with the can lying down. The robot starts in a new pose.
      </Narration>
      <Narration label="How it augments · ▶▶ 1.5×" opacity={vis(frame, t5 + q, t5 + 2 * q, 10)}>
        Your motion is re-fitted to each scene. Here it stands the can up first.
      </Narration>
      <Narration label="How it augments · ▶▶ 1.5×" opacity={vis(frame, t5 + 2 * q, t5 + 3 * q, 10)}>
        Each new scene moves the can and the target, up to 10 cm.
      </Narration>
      <Narration label="Then it checks · ▶▶ 1.5×" opacity={vis(frame, t5 + 3 * q, DEMO_FRAMES, 10)}>
        Every scene runs in physics. Only successes become training data.
      </Narration>
    </AbsoluteFill>
  );
};

/* ================================================================== HOW IT WORKS (after the demo) */
export const HowTitle: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade(12, 12);
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000", justifyContent: "center", alignItems: "center" }}>
      <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 110, color: C.text, opacity: rise(frame, 4, 22), translate: `0px ${(1 - rise(frame, 4, 22)) * 24}px` }}>
        How does it work?
      </div>
    </AbsoluteFill>
  );
};

const SLIDES: [string, string, string][] = [
  ["slide_detect_v3.png", "1 · See", "A normal webcam rebuilds your hand in 3D and spots every grab."],
  ["slide_map_v2.png", "2 · Transfer", "We copy what your hand does, so any robot arm can follow it."],
  ["slide_learn_v2.png", "3 · Practise", "The robot practises in physics until its grip really lifts the can."],
];
export const SLIDE_FRAMES = 240;

const Slides: React.FC<{ from: number; to: number }> = ({ from, to }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: "#000" }}>
      {SLIDES.slice(from, to).map(([src, label, line], i) => {
        const o = vis(frame, i * SLIDE_FRAMES, (i + 1) * SLIDE_FRAMES, 12);
        return (
          <AbsoluteFill key={src} style={{ opacity: o }}>
            <Img src={staticFile(src)} style={{ position: "absolute", left: 173, top: 24, width: 1574, height: 885 }} />
            <div style={{ position: "absolute", left: 0, right: 0, top: 930, textAlign: "center" }}>
              <div style={{ fontFamily: MONO, fontSize: 24, letterSpacing: "0.24em", textTransform: "uppercase", color: C.go }}>{label}</div>
              <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 50, color: C.text, marginTop: 8 }}>{line}</div>
            </div>
          </AbsoluteFill>
        );
      })}
    </AbsoluteFill>
  );
};
/** See + Transfer slides. */
export const SlidesSeeMap: React.FC = () => <Slides from={0} to={2} />;
/** Practise slide. */
export const SlideLearn: React.FC = () => <Slides from={2} to={3} />;

/* ---- any arm: the same hand path, followed by arms with different joints (planar CCD IK) */
const ARMS: [string, number[]][] = [
  ["7-joint arm", [0.16, 0.14, 0.13, 0.12, 0.11, 0.08, 0.06]],
  ["6-joint arm", [0.2, 0.18, 0.15, 0.12, 0.09, 0.06]],
  ["4-joint arm", [0.28, 0.24, 0.18, 0.1]],
];
const BASE: [number, number] = [0.5, 0.94];
// the hand's path in a panel (0..1, y down): reach in, lift, carry over, set down
const handPath = (u: number): [number, number] => [0.2 + 0.6 * u, 0.66 - 0.4 * Math.sin(Math.PI * u)];
const ARM_FRAMES = 270;
const uAt = (f: number) => 0.5 - 0.5 * Math.cos((2 * Math.PI * Math.max(0, f - 30)) / 200);

const chain = (L: number[], th: number[]) => {
  const pts: [number, number][] = [BASE];
  let a = 0;
  for (let i = 0; i < L.length; i++) {
    a += th[i];
    const [x, y] = pts[i];
    pts.push([x + L[i] * Math.cos(a), y + L[i] * Math.sin(a)]);
  }
  return pts;
};

/** Joint positions at frame f: FABRIK IK, warm-started frame by frame from a rest pose
 * (deterministic: every frame re-runs from frame 0). */
const solveArm = (L: number[], f: number) => {
  const rest = L.map((_, i) => (i === 0 ? -Math.PI / 2 : i % 2 ? 0.35 : -0.2));
  const p = chain(L, rest);
  const n = L.length;
  const step = (t: [number, number], iters: number) => {
    for (let it = 0; it < iters; it++) {
      p[n] = [t[0], t[1]];
      for (let i = n - 1; i >= 0; i--) {
        const dx = p[i][0] - p[i + 1][0];
        const dy = p[i][1] - p[i + 1][1];
        const m = Math.hypot(dx, dy) || 1;
        p[i] = [p[i + 1][0] + (dx / m) * L[i], p[i + 1][1] + (dy / m) * L[i]];
      }
      p[0] = [BASE[0], BASE[1]];
      for (let i = 0; i < n; i++) {
        const dx = p[i + 1][0] - p[i][0];
        const dy = p[i + 1][1] - p[i][1];
        const m = Math.hypot(dx, dy) || 1;
        p[i + 1] = [p[i][0] + (dx / m) * L[i], p[i][1] + (dy / m) * L[i]];
      }
    }
  };
  for (let k = 0; k <= f; k++) step(handPath(uAt(k)), k === 0 ? 60 : 4);
  return p;
};

const ArmPanel: React.FC<{ name: string; L: number[]; at: number }> = ({ name, L, at }) => {
  const frame = useCurrentFrame();
  const S = 470;
  const pts = solveArm(L, frame);
  const u = uAt(frame);
  const path = Array.from({ length: 61 }, (_, i) => handPath(i / 60));
  const d = (q: [number, number][]) => q.map(([x, y], i) => `${i ? "L" : "M"}${(x * S).toFixed(1)},${(y * S).toFixed(1)}`).join(" ");
  const tip = handPath(u);
  return (
    <div style={{ opacity: rise(frame, at, at + 14) }}>
      <svg width={S} height={S} style={{ background: "#0b0c0f", borderRadius: 12, border: `1.5px solid ${C.line}` }}>
        <path d={d(path)} stroke="#FF8A3D" strokeOpacity={0.35} strokeWidth={4} strokeDasharray="10 10" fill="none" />
        <rect x={BASE[0] * S - 36} y={BASE[1] * S - 4} width={72} height={26} rx={6} fill="#2a2d33" />
        <path d={d(pts)} stroke="#E8EAED" strokeWidth={16} strokeLinecap="round" strokeLinejoin="round" fill="none" />
        {pts.slice(0, -1).map(([x, y], i) => (
          <circle key={i} cx={x * S} cy={y * S} r={9} fill="#4F82FF" stroke="#0b0c0f" strokeWidth={3} />
        ))}
        <circle cx={tip[0] * S} cy={tip[1] * S} r={11} fill="#FF8A3D" />
      </svg>
      <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 36, color: C.text, marginTop: 18, textAlign: "center" }}>{name}</div>
    </div>
  );
};

export const AnyArm: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade(12, 14);
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000" }}>
      <div style={{ position: "absolute", left: 0, right: 0, top: 70, textAlign: "center" }}>
        <div style={{ fontFamily: MONO, fontSize: 28, letterSpacing: "0.26em", textTransform: "uppercase", color: C.go, opacity: rise(frame, 2, 14) }}>
          The big win
        </div>
        <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 76, color: C.text, marginTop: 10, opacity: rise(frame, 6, 20) }}>
          One recording, any robot arm.
        </div>
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 290, display: "flex", justifyContent: "center", gap: 70 }}>
        {ARMS.map(([n, L], i) => (
          <ArmPanel key={n} name={n} L={L} at={14 + i * 8} />
        ))}
      </div>
      <div style={{ position: "absolute", left: 0, right: 0, top: 880, textAlign: "center", opacity: rise(frame, 40, 54) }}>
        <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 46, color: C.text }}>
          We save <span style={{ color: "#FF8A3D" }}>the hand's path</span>, not joint angles. Each arm solves its own joints.
        </div>
        <div style={{ fontFamily: MONO, fontSize: 18, color: C.faint, marginTop: 18 }}>
          Illustration. Our demos run on a simulated Franka Panda; other arms follow the same path with their own IK.
        </div>
      </div>
    </AbsoluteFill>
  );
};
export const ANY_ARM_FRAMES = ARM_FRAMES;

/** Text column + a framed recording (scale grid, Tiger dashboard). */
const Split: React.FC<{
  kicker: string;
  title: string;
  points: string[];
  children: React.ReactNode;
  footer?: React.ReactNode;
}> = ({ kicker, title, points, children, footer }) => {
  const frame = useCurrentFrame();
  const o = useSceneFade();
  return (
    <AbsoluteFill style={{ opacity: o }}>
      <Backdrop progress={0.8} arc />
      <div style={{ position: "absolute", left: 80, top: 170, width: 640 }}>
        <Reveal at={4} dy={16}>
          <Kicker color={C.go}>{kicker}</Kicker>
        </Reveal>
        <Reveal at={10} dy={16} style={{ marginTop: 18 }}>
          <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 66, color: C.text, lineHeight: 1.08 }}>{title}</div>
        </Reveal>
        <div style={{ marginTop: 44, display: "flex", flexDirection: "column", gap: 26 }}>
          {points.map((p, i) => (
            <Reveal key={p} at={40 + i * 30} dy={12}>
              <div style={{ borderLeft: `3px solid ${C.go}`, paddingLeft: 22, fontFamily: FONT, fontWeight: 400, fontSize: 38, color: C.text }}>
                {p}
              </div>
            </Reveal>
          ))}
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 190,
          width: 1080,
          height: 640,
          borderRadius: 10,
          overflow: "hidden",
          border: `1.5px solid ${C.line}`,
          opacity: rise(frame, 6, 26),
        }}
      >
        {children}
      </div>
      {footer && <div style={{ position: "absolute", right: 80, top: 868, width: 1080, opacity: rise(frame, 120, 140) }}>{footer}</div>}
    </AbsoluteFill>
  );
};

export const ScaleExplain: React.FC = () => (
  <Split kicker="Scale" title="One demo becomes twelve." points={["New positions and poses", "Built on NVIDIA's MimicGen", "Only successes are kept"]}>
    {/* the finished grid (after the tiles fill), the app recording scaled into the frame */}
    <div style={{ position: "absolute", left: 0, top: -20, width: 1080, height: 664 }}>
      <Video src={staticFile("clips/scale_b.mp4")} trimBefore={330} objectFit="fill" muted style={{ width: "100%", height: "100%" }} />
    </div>
  </Split>
);

// each Tiger Data feature (the track's wording) and what OneGrasp does with it
const TIGER_ROWS: [string, string][] = [
  ["Unified data stack", "Demos + 30 Hz hand and robot streams, one DB"],
  ["Standard SQL", "One JOIN pulls any demo to replay it in MuJoCo"],
  ["Continuous aggregates", "Power our live, lag-free dashboard"],
  ["Columnstore compression", "Robot steps stored 1.8× smaller"],
];

export const Tiger: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade();
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000" }}>
      <div style={{ position: "absolute", left: 80, top: 175, width: 760 }}>
        <Reveal at={4} dy={16}>
          <Kicker color={C.go}>Tiger Data</Kicker>
        </Reveal>
        <Reveal at={10} dy={16} style={{ marginTop: 14 }}>
          <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 64, color: C.text, lineHeight: 1.08 }}>
            Our main database. Every demo lives here.
          </div>
        </Reveal>
        <Reveal at={24} dy={12} style={{ marginTop: 16 }}>
          <div style={{ fontFamily: FONT, fontWeight: 400, fontSize: 32, color: C.dim }}>Streamed in live, as it's recorded.</div>
        </Reveal>
        <div style={{ marginTop: 40, display: "flex", flexDirection: "column", gap: 24 }}>
          {TIGER_ROWS.map(([k, v], i) => (
            <Reveal key={k} at={50 + i * 40} dy={12}>
              <div style={{ borderLeft: `3px solid ${C.go}`, paddingLeft: 22 }}>
                <div style={{ fontFamily: MONO, fontSize: 22, letterSpacing: "0.2em", textTransform: "uppercase", color: C.go }}>{k}</div>
                <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 34, color: C.text, marginTop: 4 }}>{v}</div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 215,
          width: 960,
          height: 560,
          borderRadius: 10,
          overflow: "hidden",
          border: `1.5px solid ${C.line}`,
          opacity: rise(frame, 6, 26),
        }}
      >
        <Video src={staticFile("clips/tiger.mp4")} objectFit="cover" muted style={{ width: "100%", height: "100%" }} />
      </div>
      <div style={{ position: "absolute", right: 80, top: 815, width: 960, display: "flex", justifyContent: "space-between", opacity: rise(frame, 220, 240) }}>
        {[
          ["460+", "robot demos"],
          ["183,181", "robot steps"],
          ["30 Hz", "live hand rows"],
        ].map(([n, l]) => (
          <div key={l}>
            <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 56, color: C.text }}>{n}</div>
            <div style={{ fontFamily: MONO, fontSize: 20, letterSpacing: "0.2em", color: C.dim, textTransform: "uppercase" }}>{l}</div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

/* ================================================================== CLOSE */
export const Impact: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade(12, 14);
  const strike = rise(frame, 34, 50);
  return (
    <AbsoluteFill style={{ opacity: o }}>
      <Backdrop progress={0.95} arc />
      <AbsoluteFill style={{ justifyContent: "center", padding: "0 160px" }}>
        <div style={{ position: "relative", alignSelf: "flex-start", opacity: rise(frame, 4, 14) }}>
          <div style={{ fontFamily: FONT, fontWeight: 400, fontSize: 64, color: C.dim }}>Today: a lab and expensive gear.</div>
          <div style={{ position: "absolute", left: 0, top: "54%", height: 4, width: `${strike * 100}%`, background: C.warn }} />
        </div>
        <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 92, color: C.text, marginTop: 24, opacity: rise(frame, 54, 66) }}>
          OneGrasp: a camera.
        </div>
        <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 110, color: C.text, marginTop: 90, opacity: rise(frame, 100, 114) }}>
          Robots can learn from all of us.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

/** The team waving (phone camera), last shot. */
export const Wave: React.FC = () => {
  const o = useSceneFade(8, 16);
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000" }}>
      <Video src={staticFile("clips/wave.mp4")} objectFit="cover" muted style={{ width: "100%", height: "100%" }} />
    </AbsoluteFill>
  );
};

export const EndCard: React.FC = () => {
  const frame = useCurrentFrame();
  const o = useSceneFade(14, 20);
  return (
    <AbsoluteFill style={{ opacity: o, background: "#000", justifyContent: "center", alignItems: "center", textAlign: "center" }}>
      <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 140, letterSpacing: "0.3em", marginRight: "-0.3em", color: C.text, opacity: rise(frame, 2, 14) }}>
        ONEGRASP
      </div>
      <div style={{ fontFamily: FONT, fontWeight: 400, fontSize: 48, color: C.dim, marginTop: 20, opacity: rise(frame, 14, 26) }}>
        Teach a robot with a webcam.
      </div>
    </AbsoluteFill>
  );
};
