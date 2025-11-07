import React, { useState } from 'react';
import { useStore } from '../services/store';
import { Button } from '../components/Button';
import { calculateCost, formatTimestamp } from '../utils/helpers';
import './GenerationQueue.css';

export const GenerationQueue: React.FC = () => {
  const { currentProject, setCurrentScreen, generationJobs, startGeneration } = useStore();
  const [selectedBranches, setSelectedBranches] = useState<string[]>([]);

  if (!currentProject) return null;

  const toggleBranch = (branchId: string) => {
    setSelectedBranches((prev) =>
      prev.includes(branchId) ? prev.filter((id) => id !== branchId) : [...prev, branchId]
    );
  };

  const handleGenerate = () => {
    if (selectedBranches.length === 0) return;

    if (window.confirm(`Generate ${selectedBranches.length} branch(es)?`)) {
      startGeneration(selectedBranches);
      setSelectedBranches([]);
    }
  };

  const totalCost = selectedBranches.reduce((sum, branchId) => {
    const branch = currentProject.branches.find((b) => b.id === branchId);
    if (!branch) return sum;
    const branchDuration = branch.transitions.reduce((s, t) => s + t.duration, 0);
    return sum + calculateCost(branchDuration, currentProject.settings.defaultModel);
  }, 0);

  const totalDuration = selectedBranches.reduce((sum, branchId) => {
    const branch = currentProject.branches.find((b) => b.id === branchId);
    if (!branch) return sum;
    return sum + branch.transitions.reduce((s, t) => s + t.duration, 0);
  }, 0);

  return (
    <div className="generation-queue">
      <header className="queue-header">
        <div className="queue-header-left">
          <button className="back-btn" onClick={() => setCurrentScreen('canvas')}>
            ← Back
          </button>
          <h1>Generation Queue</h1>
        </div>
      </header>

      <div className="queue-content">
        <section className="queue-section">
          <h2>Ready to Generate</h2>

          <div className="branches-list">
            {currentProject.branches.map((branch) => {
              const duration = branch.transitions.reduce((sum, t) => sum + t.duration, 0);
              const cost = calculateCost(duration, currentProject.settings.defaultModel);
              const missingPrompts = branch.transitions.filter((t) => !t.prompt).length;
              const missingImages = branch.keyframes.filter((k) => !k.image.dataUrl).length;
              const isReady = missingPrompts === 0 && missingImages === 0;

              return (
                <div key={branch.id} className="branch-item">
                  <input
                    type="checkbox"
                    checked={selectedBranches.includes(branch.id)}
                    onChange={() => toggleBranch(branch.id)}
                    disabled={!isReady}
                  />
                  <div className="branch-info">
                    <h3>
                      {branch.name} ({duration}s)
                    </h3>
                    <p className="branch-meta">
                      {branch.transitions.length} transitions | Estimated cost: ${cost.toFixed(2)}
                    </p>
                    {isReady ? (
                      <p className="status-ready">All prompts complete ✓</p>
                    ) : (
                      <p className="status-error">
                        {missingPrompts > 0 && `Missing ${missingPrompts} prompts`}
                        {missingPrompts > 0 && missingImages > 0 && ' | '}
                        {missingImages > 0 && `Missing ${missingImages} images`} ✗
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="generation-settings">
            <h3>Generation Settings</h3>
            <div className="settings-grid">
              <div className="setting-item">
                <label>Model:</label>
                <span>{currentProject.settings.defaultModel === 'veo3_fast' ? 'Veo3 Fast' : 'Veo3 Quality'}</span>
              </div>
              <div className="setting-item">
                <label>FPS:</label>
                <span>{currentProject.settings.defaultFPS}</span>
              </div>
              <div className="setting-item">
                <label>Resolution:</label>
                <span>{currentProject.settings.defaultResolution}</span>
              </div>
            </div>
          </div>

          {selectedBranches.length > 0 && (
            <div className="generation-summary">
              <h3>Summary</h3>
              <p>
                Total selected: {selectedBranches.length} branch(es) ({totalDuration} seconds)
              </p>
              <p>Estimated cost: ${totalCost.toFixed(2)}</p>
              <p>Estimated time: ~{Math.ceil(totalDuration / 3)} minutes</p>
            </div>
          )}

          <div className="queue-actions">
            <Button
              onClick={() => setSelectedBranches(currentProject.branches.map((b) => b.id))}
              variant="secondary"
            >
              Select All
            </Button>
            <Button onClick={() => setSelectedBranches([])} variant="secondary">
              Deselect All
            </Button>
            <Button onClick={handleGenerate} disabled={selectedBranches.length === 0}>
              Generate Selected
            </Button>
          </div>
        </section>

        <section className="queue-section">
          <h2>Generation History</h2>
          {generationJobs.length === 0 ? (
            <p className="empty-message">No generation history yet</p>
          ) : (
            <div className="history-list">
              {generationJobs.map((job) => (
                <div key={job.id} className="history-item">
                  <div className={`status-dot status-${job.status}`} />
                  <div className="history-info">
                    <h4>
                      {job.branchName} - {formatTimestamp(job.createdAt)}
                    </h4>
                    <p>
                      {job.status === 'complete' && `✓ Complete (${job.totalTransitions} transitions)`}
                      {job.status === 'in_progress' &&
                        `In progress... ${job.completedTransitions}/${job.totalTransitions}`}
                      {job.status === 'failed' && `✗ Failed: ${job.error}`}
                      {job.status === 'pending' && 'Pending...'}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};
