import { HeadPose } from "./facePose";
import { Scores } from "./blendshapeEmotion";
import { ParameterEngine } from "./parameterEngine";

export function calculateDeepTechScore(
  pose: HeadPose,
  scores: Scores,
  area: number,
  engine: ParameterEngine
): number {
  // Construct feature vector from all available data
  const features = [
    pose.yaw,
    pose.pitch,
    pose.roll,
    pose.distance,
    area,
    ...Object.values(scores),
  ];

  // Use the first block of parameters for the Deep Tech modulation
  return engine.modulate(features, 0);
}
