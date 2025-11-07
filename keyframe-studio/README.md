# Keyframe Studio

**Version:** 1.0
**Purpose:** Generate synthetic game training data through keyframe interpolation

## Overview

Keyframe Studio allows you to create comprehensive game training datasets by:
1. Defining game action spaces
2. Creating keyframe sequences (visual milestones)
3. Writing cinematic prompts between keyframes
4. Generating branching gameplay paths
5. Exporting complete training datasets

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Build

```bash
npm run build
```

Built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Features

### 1. Projects Dashboard
- Create and manage multiple game projects
- View project statistics and generation status
- Search and filter projects
- Duplicate or delete projects

### 2. Action Space Editor
- Define game-specific actions (MOVE_FORWARD, JUMP, etc.)
- Create action combinations (MOVE_FORWARD + JUMP)
- Validate action space before generation

### 3. Keyframe Canvas (Main Editor)
- Visual timeline showing all keyframes
- Upload reference images for each keyframe
- Set timestamps and actions
- Edit prompts between keyframes
- Create branching paths

### 4. Generation Queue
- Select branches for video generation
- Configure generation settings (model, FPS, resolution)
- View cost estimates
- Monitor generation progress
- View generation history

### 5. Dataset Export
- Export generated videos and annotations
- Multiple format options:
  - Training Dataset Package (videos + actions + metadata)
  - Raw Videos Only
  - Action Logs Only
- Organize exports in flat or structured format

## Project Structure

```
keyframe-studio/
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Modal.tsx
│   │   └── ToastContainer.tsx
│   ├── screens/            # Main application screens
│   │   ├── ProjectsDashboard.tsx
│   │   ├── ActionSpaceEditor.tsx
│   │   ├── KeyframeCanvas.tsx
│   │   ├── GenerationQueue.tsx
│   │   └── DatasetExport.tsx
│   ├── services/           # State management
│   │   └── store.ts
│   ├── types/              # TypeScript type definitions
│   │   └── index.ts
│   ├── utils/              # Helper functions
│   │   └── helpers.ts
│   ├── styles/             # Global styles
│   │   └── global.css
│   ├── App.tsx             # Main app component
│   └── main.tsx            # Entry point
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Data Storage

All project data is stored in the browser's LocalStorage under the key `keyframe-studio-projects`. This allows for persistent data across sessions without requiring a backend.

## Keyboard Shortcuts

- `Escape` - Close modal/cancel action
- `Enter` - Submit form (when applicable)

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Zustand** - State management
- **Vite** - Build tool
- **CSS3** - Styling

## Future Enhancements

- Collaborative editing (multi-user support)
- Cloud sync
- Version control (Git-like branching)
- Pre-made project templates
- Shared asset library
- Batch operations
- Real-time animation preview
- Style presets
- Cost optimizer
- Quality analyzer

## License

MIT
