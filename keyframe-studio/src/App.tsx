import React, { useEffect } from 'react';
import { useStore } from './services/store';
import { ProjectsDashboard } from './screens/ProjectsDashboard';
import { ActionSpaceEditor } from './screens/ActionSpaceEditor';
import { KeyframeCanvas } from './screens/KeyframeCanvas';
import { GenerationQueue } from './screens/GenerationQueue';
import { DatasetExport } from './screens/DatasetExport';
import { ToastContainer } from './components/ToastContainer';
import './styles/global.css';

export const App: React.FC = () => {
  const { currentScreen, loadFromLocalStorage } = useStore();

  useEffect(() => {
    loadFromLocalStorage();
  }, [loadFromLocalStorage]);

  return (
    <>
      {currentScreen === 'dashboard' && <ProjectsDashboard />}
      {currentScreen === 'action-space' && <ActionSpaceEditor />}
      {currentScreen === 'canvas' && <KeyframeCanvas />}
      {currentScreen === 'generation' && <GenerationQueue />}
      {currentScreen === 'export' && <DatasetExport />}
      <ToastContainer />
    </>
  );
};
