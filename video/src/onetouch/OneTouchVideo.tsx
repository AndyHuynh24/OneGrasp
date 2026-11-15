import React from "react";
import { AbsoluteFill, Sequence, Series, useCurrentFrame } from "remotion";
import { C, StageBar } from "./ui";
import {
  DataGap,
  Demo,
  DEMO_FRAMES,
  EndCard,
  AnyArm,
  ANY_ARM_FRAMES,
  SlideLearn,
  SlidesSeeMap,
  HowTitle,
  Impact,
  Intro,
  LiveCard,
  Problem,
  SLIDE_FRAMES,
  Tiger,
  Wave,
} from "./scenes";

// [component, frames at 30 fps, stage lit in the top bar, top bar shown]
// Order: the problem (real footage) -> the demo, one continuous run -> how it works -> close.
// The top bar only shows where nothing sits under it (not over footage, recordings or slides).
export const SCENES: [React.FC, number, number, boolean][] = [
  [Problem, 270, 0, false],
  [DataGap, 240, 0, false],
  [Intro, 180, 1, false],
  [LiveCard, 90, 1, false],
  [Demo, DEMO_FRAMES, 1, false],
  [HowTitle, 75, 2, false],
  [SlidesSeeMap, 2 * SLIDE_FRAMES, 2, false],
  [AnyArm, ANY_ARM_FRAMES, 2, false],
  [SlideLearn, SLIDE_FRAMES, 2, false],
  [Tiger, 540, 3, true],
  [Impact, 230, 4, true],
  [EndCard, 110, 4, false],
  [Wave, 96, 4, false],
];
export const TOTAL = SCENES.reduce((s, [, d]) => s + d, 0);

const Bar: React.FC = () => {
  const frame = useCurrentFrame();
  let t = 0;
  let stage = 0;
  let op = 0;
  SCENES.forEach(([, d, s, show], i) => {
    if (frame >= t && frame < t + d && show) {
      stage = s;
      // fade only where the bar appears / disappears (next to a scene without it)
      const fadeIn = i === 0 || !SCENES[i - 1][3];
      const fadeOut = i === SCENES.length - 1 || !SCENES[i + 1][3];
      op = Math.min(1, fadeIn ? (frame - t) / 12 : 1, fadeOut ? (t + d - frame) / 12 : 1);
    }
    t += d;
  });
  return <StageBar stage={stage} opacity={op} />;
};

export const OneTouchVideo: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Series>
      {SCENES.map(([Comp, d], i) => (
        <Series.Sequence key={i} durationInFrames={d}>
          <Comp />
        </Series.Sequence>
      ))}
    </Series>
    <Sequence>
      <Bar />
    </Sequence>
  </AbsoluteFill>
);
