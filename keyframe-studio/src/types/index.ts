// Core data types for Keyframe Studio

export interface Project {
  id: string;
  name: string;
  created: string;
  modified: string;
  actionSpace: ActionSpace;
  branches: Branch[];
  settings: ProjectSettings;
}

export interface ActionSpace {
  coreActions: string[];
  combinations: string[];
}

export interface Branch {
  id: string;
  name: string;
  keyframes: Keyframe[];
  transitions: Transition[];
  parentBranchId?: string; // ID of the parent branch (if this is a diverging branch)
  branchPointKeyframeId?: string; // ID of the keyframe where this branch diverges
}

export interface Keyframe {
  id: string;
  branchId: string;
  timestamp: number;
  image: {
    path: string;
    dataUrl?: string; // For client-side preview
    width: number;
    height: number;
  };
  action: string;
  uiState: Record<string, string>;
}

export interface Transition {
  id: string;
  sourceKeyframe: string;
  destKeyframe: string;
  duration: number;
  action: string;
  prompt: string;
  generationStatus: 'pending' | 'in_progress' | 'complete' | 'failed';
  outputVideo?: string;
  costActual?: number;
  errorMessage?: string;
}

export interface ProjectSettings {
  defaultModel: 'veo3_fast' | 'veo3_quality';
  defaultFPS: number;
  defaultResolution: '1280x720' | '1920x1080';
}

export interface GenerationJob {
  id: string;
  branchId: string;
  branchName: string;
  status: 'pending' | 'in_progress' | 'complete' | 'failed';
  progress: number; // 0-100
  currentTransition?: string;
  totalTransitions: number;
  completedTransitions: number;
  estimatedTimeRemaining?: number; // in seconds
  error?: string;
  createdAt: string;
  completedAt?: string;
}

export interface ExportConfig {
  branches: string[];
  format: 'training_package' | 'videos_only' | 'actions_only';
  structure: 'flat' | 'organized';
  include: {
    actionAnnotations: boolean;
    keyframeImages: boolean;
    promptTexts: boolean;
    metadata: boolean;
    generationSettings: boolean;
  };
}

export type ValidationError = {
  field: string;
  message: string;
};

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  action?: {
    label: string;
    callback: () => void;
  };
  duration?: number; // in ms, undefined = stays until dismissed
}
