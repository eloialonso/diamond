import { create } from 'zustand';
import { Project, Branch, Keyframe, Transition, Toast, GenerationJob } from '../types';
import { generateId } from '../utils/helpers';

interface AppState {
  // Projects
  projects: Project[];
  currentProject: Project | null;
  currentBranch: Branch | null;

  // UI State
  currentScreen: 'dashboard' | 'action-space' | 'canvas' | 'generation' | 'export';
  selectedKeyframe: Keyframe | null;
  selectedTransition: { source: Keyframe; dest: Keyframe } | null;
  toasts: Toast[];

  // Generation
  generationJobs: GenerationJob[];

  // Actions - Projects
  createProject: (name: string) => void;
  deleteProject: (id: string) => void;
  duplicateProject: (id: string) => void;
  loadProject: (id: string) => void;
  updateProjectName: (id: string, name: string) => void;

  // Actions - Branches
  createBranch: (name: string, fromKeyframeId?: string) => void;
  deleteBranch: (branchId: string) => void;
  switchBranch: (branchId: string) => void;

  // Actions - Keyframes
  addKeyframe: (timestamp: number, imagePath: string, action: string) => void;
  updateKeyframe: (keyframeId: string, updates: Partial<Keyframe>) => void;
  deleteKeyframe: (keyframeId: string) => void;
  selectKeyframe: (keyframe: Keyframe | null) => void;

  // Actions - Transitions
  updateTransition: (sourceId: string, destId: string, updates: Partial<Transition>) => void;
  selectTransition: (source: Keyframe, dest: Keyframe) => void;

  // Actions - Action Space
  addCoreAction: (action: string) => void;
  editCoreAction: (oldAction: string, newAction: string) => void;
  deleteCoreAction: (action: string) => void;
  addActionCombination: (combination: string) => void;
  deleteActionCombination: (combination: string) => void;

  // Actions - UI
  setCurrentScreen: (screen: AppState['currentScreen']) => void;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;

  // Actions - Generation
  startGeneration: (branchIds: string[]) => void;
  updateGenerationJob: (jobId: string, updates: Partial<GenerationJob>) => void;

  // Actions - Storage
  saveToLocalStorage: () => void;
  loadFromLocalStorage: () => void;
}

// Helper to create default project
const createDefaultProject = (name: string): Project => {
  const projectId = generateId();
  const branchId = generateId();
  const kf1Id = generateId();
  const kf2Id = generateId();

  return {
    id: projectId,
    name,
    created: new Date().toISOString(),
    modified: new Date().toISOString(),
    actionSpace: {
      coreActions: [],
      combinations: []
    },
    branches: [{
      id: branchId,
      name: 'Branch 1',
      keyframes: [
        {
          id: kf1Id,
          branchId,
          timestamp: 0,
          image: { path: '', width: 1280, height: 720 },
          action: '',
          uiState: {}
        },
        {
          id: kf2Id,
          branchId,
          timestamp: 10,
          image: { path: '', width: 1280, height: 720 },
          action: '',
          uiState: {}
        }
      ],
      transitions: [{
        id: generateId(),
        sourceKeyframe: kf1Id,
        destKeyframe: kf2Id,
        duration: 10,
        action: '',
        prompt: '',
        generationStatus: 'pending'
      }]
    }],
    settings: {
      defaultModel: 'veo3_fast',
      defaultFPS: 30,
      defaultResolution: '1280x720'
    }
  };
};

export const useStore = create<AppState>((set, get) => ({
  // Initial state
  projects: [],
  currentProject: null,
  currentBranch: null,
  currentScreen: 'dashboard',
  selectedKeyframe: null,
  selectedTransition: null,
  toasts: [],
  generationJobs: [],

  // Project actions
  createProject: (name: string) => {
    const project = createDefaultProject(name);
    set((state) => ({
      projects: [...state.projects, project],
      currentProject: project,
      currentBranch: project.branches[0],
      currentScreen: 'action-space' // Start with action space editor
    }));
    get().saveToLocalStorage();
  },

  deleteProject: (id: string) => {
    set((state) => ({
      projects: state.projects.filter(p => p.id !== id),
      currentProject: state.currentProject?.id === id ? null : state.currentProject
    }));
    get().saveToLocalStorage();
  },

  duplicateProject: (id: string) => {
    const original = get().projects.find(p => p.id === id);
    if (!original) return;

    const duplicate: Project = {
      ...original,
      id: generateId(),
      name: `${original.name} - Copy`,
      created: new Date().toISOString(),
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: [...state.projects, duplicate]
    }));
    get().saveToLocalStorage();
  },

  loadProject: (id: string) => {
    const project = get().projects.find(p => p.id === id);
    if (project) {
      set({
        currentProject: project,
        currentBranch: project.branches[0] || null,
        currentScreen: 'canvas'
      });
    }
  },

  updateProjectName: (id: string, name: string) => {
    set((state) => ({
      projects: state.projects.map(p =>
        p.id === id ? { ...p, name, modified: new Date().toISOString() } : p
      ),
      currentProject: state.currentProject?.id === id
        ? { ...state.currentProject, name, modified: new Date().toISOString() }
        : state.currentProject
    }));
    get().saveToLocalStorage();
  },

  // Branch actions
  createBranch: (name: string, fromKeyframeId?: string) => {
    const { currentProject, currentBranch } = get();
    if (!currentProject) return;

    const branchId = generateId();
    let newBranch: Branch;

    if (fromKeyframeId && currentBranch) {
      // Creating a diverging branch from a specific keyframe
      const branchPointKeyframe = currentBranch.keyframes.find(k => k.id === fromKeyframeId);
      if (!branchPointKeyframe) return;

      // Create branch with parent information
      newBranch = {
        id: branchId,
        name,
        parentBranchId: currentBranch.id,
        branchPointKeyframeId: fromKeyframeId,
        keyframes: [{
          id: generateId(),
          branchId,
          timestamp: branchPointKeyframe.timestamp + 10, // Start 10s after branch point
          image: { path: '', width: 1280, height: 720 },
          action: '',
          uiState: {}
        }],
        transitions: []
      };
    } else {
      // Creating an independent root branch
      newBranch = {
        id: branchId,
        name,
        keyframes: [{
          id: generateId(),
          branchId,
          timestamp: 0,
          image: { path: '', width: 1280, height: 720 },
          action: '',
          uiState: {}
        }],
        transitions: []
      };
    }

    const updatedProject = {
      ...currentProject,
      branches: [...currentProject.branches, newBranch],
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject,
      currentBranch: newBranch
    }));
    get().saveToLocalStorage();
  },

  deleteBranch: (branchId: string) => {
    const { currentProject } = get();
    if (!currentProject) return;

    const updatedProject = {
      ...currentProject,
      branches: currentProject.branches.filter(b => b.id !== branchId),
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject,
      currentBranch: state.currentBranch?.id === branchId ? updatedProject.branches[0] : state.currentBranch
    }));
    get().saveToLocalStorage();
  },

  switchBranch: (branchId: string) => {
    const { currentProject } = get();
    if (!currentProject) return;

    const branch = currentProject.branches.find(b => b.id === branchId);
    if (branch) {
      set({ currentBranch: branch });
    }
  },

  // Keyframe actions
  addKeyframe: (timestamp: number, imagePath: string, action: string) => {
    const { currentProject, currentBranch } = get();
    if (!currentProject || !currentBranch) return;

    const keyframeId = generateId();
    const newKeyframe: Keyframe = {
      id: keyframeId,
      branchId: currentBranch.id,
      timestamp,
      image: { path: imagePath, width: 1280, height: 720 },
      action,
      uiState: {}
    };

    // Add keyframe and create transition from previous keyframe
    const sortedKeyframes = [...currentBranch.keyframes, newKeyframe].sort((a, b) => a.timestamp - b.timestamp);
    const keyframeIndex = sortedKeyframes.findIndex(kf => kf.id === keyframeId);

    let newTransitions = [...currentBranch.transitions];

    if (keyframeIndex > 0) {
      const prevKeyframe = sortedKeyframes[keyframeIndex - 1];
      newTransitions.push({
        id: generateId(),
        sourceKeyframe: prevKeyframe.id,
        destKeyframe: keyframeId,
        duration: timestamp - prevKeyframe.timestamp,
        action,
        prompt: '',
        generationStatus: 'pending'
      });
    }

    if (keyframeIndex < sortedKeyframes.length - 1) {
      const nextKeyframe = sortedKeyframes[keyframeIndex + 1];
      // Remove old transition to next keyframe if it exists
      newTransitions = newTransitions.filter(t =>
        !(t.destKeyframe === nextKeyframe.id && keyframeIndex > 0 && t.sourceKeyframe === sortedKeyframes[keyframeIndex - 1].id)
      );
      newTransitions.push({
        id: generateId(),
        sourceKeyframe: keyframeId,
        destKeyframe: nextKeyframe.id,
        duration: nextKeyframe.timestamp - timestamp,
        action: nextKeyframe.action,
        prompt: '',
        generationStatus: 'pending'
      });
    }

    const updatedBranch = {
      ...currentBranch,
      keyframes: sortedKeyframes,
      transitions: newTransitions
    };

    const updatedProject = {
      ...currentProject,
      branches: currentProject.branches.map(b => b.id === currentBranch.id ? updatedBranch : b),
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject,
      currentBranch: updatedBranch
    }));
    get().saveToLocalStorage();
  },

  updateKeyframe: (keyframeId: string, updates: Partial<Keyframe>) => {
    const { currentProject, currentBranch } = get();
    if (!currentProject || !currentBranch) return;

    const updatedBranch = {
      ...currentBranch,
      keyframes: currentBranch.keyframes.map(kf =>
        kf.id === keyframeId ? { ...kf, ...updates } : kf
      )
    };

    const updatedProject = {
      ...currentProject,
      branches: currentProject.branches.map(b => b.id === currentBranch.id ? updatedBranch : b),
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject,
      currentBranch: updatedBranch,
      selectedKeyframe: state.selectedKeyframe?.id === keyframeId ? { ...state.selectedKeyframe, ...updates } : state.selectedKeyframe
    }));
    get().saveToLocalStorage();
  },

  deleteKeyframe: (keyframeId: string) => {
    const { currentProject, currentBranch } = get();
    if (!currentProject || !currentBranch) return;

    const updatedBranch = {
      ...currentBranch,
      keyframes: currentBranch.keyframes.filter(kf => kf.id !== keyframeId),
      transitions: currentBranch.transitions.filter(t =>
        t.sourceKeyframe !== keyframeId && t.destKeyframe !== keyframeId
      )
    };

    const updatedProject = {
      ...currentProject,
      branches: currentProject.branches.map(b => b.id === currentBranch.id ? updatedBranch : b),
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject,
      currentBranch: updatedBranch,
      selectedKeyframe: state.selectedKeyframe?.id === keyframeId ? null : state.selectedKeyframe
    }));
    get().saveToLocalStorage();
  },

  selectKeyframe: (keyframe: Keyframe | null) => {
    set({ selectedKeyframe: keyframe, selectedTransition: null });
  },

  // Transition actions
  updateTransition: (sourceId: string, destId: string, updates: Partial<Transition>) => {
    const { currentProject, currentBranch } = get();
    if (!currentProject || !currentBranch) return;

    const updatedBranch = {
      ...currentBranch,
      transitions: currentBranch.transitions.map(t =>
        t.sourceKeyframe === sourceId && t.destKeyframe === destId ? { ...t, ...updates } : t
      )
    };

    const updatedProject = {
      ...currentProject,
      branches: currentProject.branches.map(b => b.id === currentBranch.id ? updatedBranch : b),
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject,
      currentBranch: updatedBranch
    }));
    get().saveToLocalStorage();
  },

  selectTransition: (source: Keyframe, dest: Keyframe) => {
    set({ selectedTransition: { source, dest }, selectedKeyframe: null });
  },

  // Action Space actions
  addCoreAction: (action: string) => {
    const { currentProject } = get();
    if (!currentProject) return;

    const updatedProject = {
      ...currentProject,
      actionSpace: {
        ...currentProject.actionSpace,
        coreActions: [...currentProject.actionSpace.coreActions, action]
      },
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject
    }));
    get().saveToLocalStorage();
  },

  editCoreAction: (oldAction: string, newAction: string) => {
    const { currentProject } = get();
    if (!currentProject) return;

    const updatedProject = {
      ...currentProject,
      actionSpace: {
        coreActions: currentProject.actionSpace.coreActions.map(a => a === oldAction ? newAction : a),
        combinations: currentProject.actionSpace.combinations.map(c =>
          c.replace(oldAction, newAction)
        )
      },
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject
    }));
    get().saveToLocalStorage();
  },

  deleteCoreAction: (action: string) => {
    const { currentProject } = get();
    if (!currentProject) return;

    const updatedProject = {
      ...currentProject,
      actionSpace: {
        coreActions: currentProject.actionSpace.coreActions.filter(a => a !== action),
        combinations: currentProject.actionSpace.combinations.filter(c => !c.includes(action))
      },
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject
    }));
    get().saveToLocalStorage();
  },

  addActionCombination: (combination: string) => {
    const { currentProject } = get();
    if (!currentProject) return;

    const updatedProject = {
      ...currentProject,
      actionSpace: {
        ...currentProject.actionSpace,
        combinations: [...currentProject.actionSpace.combinations, combination]
      },
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject
    }));
    get().saveToLocalStorage();
  },

  deleteActionCombination: (combination: string) => {
    const { currentProject } = get();
    if (!currentProject) return;

    const updatedProject = {
      ...currentProject,
      actionSpace: {
        ...currentProject.actionSpace,
        combinations: currentProject.actionSpace.combinations.filter(c => c !== combination)
      },
      modified: new Date().toISOString()
    };

    set((state) => ({
      projects: state.projects.map(p => p.id === currentProject.id ? updatedProject : p),
      currentProject: updatedProject
    }));
    get().saveToLocalStorage();
  },

  // UI actions
  setCurrentScreen: (screen: AppState['currentScreen']) => {
    set({ currentScreen: screen });
  },

  addToast: (toast: Omit<Toast, 'id'>) => {
    const newToast: Toast = { ...toast, id: generateId() };
    set((state) => ({
      toasts: [...state.toasts, newToast].slice(-3) // Keep max 3 toasts
    }));

    if (toast.duration !== undefined) {
      setTimeout(() => {
        get().removeToast(newToast.id);
      }, toast.duration);
    }
  },

  removeToast: (id: string) => {
    set((state) => ({
      toasts: state.toasts.filter(t => t.id !== id)
    }));
  },

  // Generation actions
  startGeneration: (branchIds: string[]) => {
    const { currentProject } = get();
    if (!currentProject) return;

    const newJobs = branchIds.map(branchId => {
      const branch = currentProject.branches.find(b => b.id === branchId);
      if (!branch) return null;

      return {
        id: generateId(),
        branchId,
        branchName: branch.name,
        status: 'pending' as const,
        progress: 0,
        totalTransitions: branch.transitions.length,
        completedTransitions: 0,
        createdAt: new Date().toISOString()
      };
    }).filter(Boolean) as GenerationJob[];

    set((state) => ({
      generationJobs: [...state.generationJobs, ...newJobs]
    }));

    get().addToast({
      type: 'info',
      message: 'Generation started',
      duration: 2000
    });
  },

  updateGenerationJob: (jobId: string, updates: Partial<GenerationJob>) => {
    set((state) => ({
      generationJobs: state.generationJobs.map(j =>
        j.id === jobId ? { ...j, ...updates } : j
      )
    }));
  },

  // Storage actions
  saveToLocalStorage: () => {
    const { projects } = get();
    try {
      localStorage.setItem('keyframe-studio-projects', JSON.stringify(projects));
    } catch (error) {
      console.error('Failed to save to localStorage:', error);
    }
  },

  loadFromLocalStorage: () => {
    try {
      const stored = localStorage.getItem('keyframe-studio-projects');
      if (stored) {
        const projects = JSON.parse(stored);
        set({ projects });
      }
    } catch (error) {
      console.error('Failed to load from localStorage:', error);
    }
  }
}));
