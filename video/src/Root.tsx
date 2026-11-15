import { Composition } from "remotion";
import { OneTouchVideo, TOTAL } from "./onetouch/OneTouchVideo";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="OneTouch"
      component={OneTouchVideo}
      durationInFrames={TOTAL}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
