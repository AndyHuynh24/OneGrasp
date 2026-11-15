import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { loadFont as loadBarlow } from "@remotion/google-fonts/Barlow";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

const barlow = loadBarlow("normal", {
  weights: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
});
const mono = loadMono("normal", { weights: ["400", "500"], subsets: ["latin"] });

export const FONT = barlow.fontFamily;
export const MONO = mono.fontFamily;
export const C = {
  bg: "#000000",
  text: "#FFFFFF",
  dim: "rgba(255,255,255,0.62)",
  faint: "rgba(255,255,255,0.28)",
  line: "rgba(255,255,255,0.22)",
  go: "#7CE38B",
  warn: "#FF6B5E",
};

const EASE = Easing.bezier(0.16, 1, 0.3, 1);
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

/** 0 -> 1 between frames a and b (eased). */
export const rise = (frame: number, a: number, b: number) =>
  interpolate(frame, [a, b], [0, 1], { ...clamp, easing: EASE });

/** Opacity for a scene: fade in over `inF` frames, out over the last `outF`. */
export const useSceneFade = (inF = 12, outF = 12) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  return Math.min(
    interpolate(frame, [0, inF], [0, 1], clamp),
    interpolate(frame, [durationInFrames - outF, durationInFrames], [1, 0], clamp),
  );
};

/** Visible (fade in/out) between frames `from` and `to` of the current sequence. */
export const vis = (frame: number, from: number, to: number, f = 10) =>
  Math.min(
    interpolate(frame, [from, from + f], [0, 1], clamp),
    interpolate(frame, [to - f, to], [1, 0], clamp),
  );

export const Kicker: React.FC<{ children: React.ReactNode; color?: string; style?: React.CSSProperties }> = ({
  children,
  color = C.dim,
  style,
}) => (
  <div
    style={{
      fontFamily: MONO,
      fontSize: 26,
      letterSpacing: "0.32em",
      textTransform: "uppercase",
      color,
      ...style,
    }}
  >
    {children}
  </div>
);

/** A headline that slides up and fades in, starting at frame `at`. */
export const Reveal: React.FC<{
  at?: number;
  children: React.ReactNode;
  style?: React.CSSProperties;
  dy?: number;
}> = ({ at = 0, children, style, dy = 40 }) => {
  const frame = useCurrentFrame();
  const p = rise(frame, at, at + 22);
  return (
    <div style={{ opacity: p, translate: `0px ${(1 - p) * dy}px`, ...style }}>{children}</div>
  );
};

/** Thin corner brackets framing the picture (webcast look). */
export const Brackets: React.FC<{ inset?: number; size?: number; opacity?: number }> = ({
  inset = 44,
  size = 46,
  opacity = 0.55,
}) => {
  const s: React.CSSProperties = { position: "absolute", width: size, height: size, opacity };
  const b = `2px solid ${C.text}`;
  return (
    <AbsoluteFill style={{ pointerEvents: "none" }}>
      <div style={{ ...s, left: inset, top: inset, borderLeft: b, borderTop: b }} />
      <div style={{ ...s, right: inset, top: inset, borderRight: b, borderTop: b }} />
      <div style={{ ...s, left: inset, bottom: inset, borderLeft: b, borderBottom: b }} />
      <div style={{ ...s, right: inset, bottom: inset, borderRight: b, borderBottom: b }} />
    </AbsoluteFill>
  );
};

export const STAGES = ["Problem", "Demo", "How it works", "Tiger Data", "Impact"] as const;

/** Top bar: wordmark + the mission stages, the current one lit. */
export const StageBar: React.FC<{ stage: number; opacity?: number }> = ({ stage, opacity = 1 }) => (
  <div
    style={{
      position: "absolute",
      top: 0,
      left: 0,
      right: 0,
      height: 92,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 80px",
      opacity,
      background: "linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0))",
    }}
  >
    <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 26, letterSpacing: "0.45em", color: C.text }}>
      ONEGRASP
    </div>
    <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
      {STAGES.map((s, i) => (
        <React.Fragment key={s}>
          {i > 0 && <div style={{ width: 54, height: 1, background: i <= stage ? C.text : C.line }} />}
          <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "0 6px" }}>
            <div
              style={{
                width: 12,
                height: 12,
                borderRadius: 6,
                border: `2px solid ${i <= stage ? C.text : C.faint}`,
                background: i === stage ? C.text : "transparent",
              }}
            />
            <div
              style={{
                fontFamily: MONO,
                fontSize: 19,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: i === stage ? C.text : C.faint,
              }}
            >
              {s}
            </div>
          </div>
        </React.Fragment>
      ))}
    </div>
  </div>
);

/** Lower-third caption: a mono label above a line of text. */
export const LowerThird: React.FC<{
  label?: string;
  text: React.ReactNode;
  opacity: number;
  labelColor?: string;
  bottom?: number;
}> = ({ label, text, opacity, labelColor = C.go, bottom = 96 }) => (
  <div
    style={{
      position: "absolute",
      left: 80,
      bottom,
      opacity,
      translate: `${(1 - opacity) * -30}px 0px`,
      display: "flex",
      alignItems: "stretch",
      gap: 26,
      background: "rgba(0,0,0,0.72)",
      padding: "22px 34px 24px 0",
      borderRadius: 4,
    }}
  >
    <div style={{ width: 5, background: labelColor }} />
    <div>
      {label && (
        <div
          style={{
            fontFamily: MONO,
            fontSize: 22,
            letterSpacing: "0.3em",
            textTransform: "uppercase",
            color: labelColor,
            marginBottom: 8,
          }}
        >
          {label}
        </div>
      )}
      <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 46, color: C.text, lineHeight: 1.15 }}>{text}</div>
    </div>
  </div>
);

/** Small rounded badge (e.g. the fast-forward speed). */
export const Badge: React.FC<{ children: React.ReactNode; style?: React.CSSProperties; color?: string }> = ({
  children,
  style,
  color = C.text,
}) => (
  <div
    style={{
      fontFamily: MONO,
      fontSize: 22,
      letterSpacing: "0.22em",
      textTransform: "uppercase",
      color,
      border: `1.5px solid ${color}`,
      borderRadius: 999,
      padding: "8px 20px",
      background: "rgba(0,0,0,0.6)",
      ...style,
    }}
  >
    {children}
  </div>
);
