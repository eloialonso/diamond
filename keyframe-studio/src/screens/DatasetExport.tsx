import React, { useState } from 'react';
import { useStore } from '../services/store';
import { Button } from '../components/Button';
import './DatasetExport.css';

export const DatasetExport: React.FC = () => {
  const { currentProject, setCurrentScreen } = useStore();
  const [selectedBranches, setSelectedBranches] = useState<string[]>([]);
  const [format, setFormat] = useState<'training_package' | 'videos_only' | 'actions_only'>('training_package');
  const [structure, setStructure] = useState<'flat' | 'organized'>('organized');
  const [include, setInclude] = useState({
    actionAnnotations: true,
    keyframeImages: true,
    promptTexts: true,
    metadata: true,
    generationSettings: false
  });

  if (!currentProject) return null;

  const toggleBranch = (branchId: string) => {
    setSelectedBranches((prev) =>
      prev.includes(branchId) ? prev.filter((id) => id !== branchId) : [...prev, branchId]
    );
  };

  const handleExport = () => {
    if (selectedBranches.length === 0) {
      alert('Please select at least one branch to export');
      return;
    }

    // This is a mock export - in a real implementation, this would:
    // 1. Gather all selected data
    // 2. Package it according to format and structure settings
    // 3. Download as a zip file or upload to cloud storage

    alert(`Exporting ${selectedBranches.length} branch(es) as ${format}...`);
  };

  const totalSize = selectedBranches.length * 150; // Mock: ~150MB per branch

  return (
    <div className="dataset-export">
      <header className="export-header">
        <div className="export-header-left">
          <button className="back-btn" onClick={() => setCurrentScreen('canvas')}>
            ← Back
          </button>
          <h1>Export Dataset</h1>
        </div>
      </header>

      <div className="export-content">
        <p className="export-project-name">Project: {currentProject.name}</p>

        <section className="export-section">
          <h2>Select Branches to Export</h2>
          <div className="branches-list">
            {currentProject.branches.map((branch) => {
              const duration = branch.transitions.reduce((sum, t) => sum + t.duration, 0);
              const generatedCount = branch.transitions.filter((t) => t.generationStatus === 'complete').length;
              const isGenerated = generatedCount === branch.transitions.length;

              return (
                <label key={branch.id} className={`branch-checkbox ${!isGenerated ? 'disabled' : ''}`}>
                  <input
                    type="checkbox"
                    checked={selectedBranches.includes(branch.id)}
                    onChange={() => toggleBranch(branch.id)}
                    disabled={!isGenerated}
                  />
                  <div className="branch-label">
                    <span className="branch-name">
                      {branch.name} ({duration}s)
                    </span>
                    {isGenerated ? (
                      <span className="branch-status generated">✓ Generated</span>
                    ) : (
                      <span className="branch-status not-generated">✗ Not generated</span>
                    )}
                  </div>
                </label>
              );
            })}
          </div>
        </section>

        <section className="export-section">
          <h2>Export Format</h2>
          <div className="radio-group">
            <label className="radio-label">
              <input
                type="radio"
                name="format"
                value="training_package"
                checked={format === 'training_package'}
                onChange={(e) => setFormat(e.target.value as any)}
              />
              <div className="radio-content">
                <strong>Training Dataset Package (.zip)</strong>
                <p>Includes: videos, action logs, metadata JSON</p>
              </div>
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name="format"
                value="videos_only"
                checked={format === 'videos_only'}
                onChange={(e) => setFormat(e.target.value as any)}
              />
              <div className="radio-content">
                <strong>Raw Videos Only (.mp4)</strong>
                <p>Just the generated video files</p>
              </div>
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name="format"
                value="actions_only"
                checked={format === 'actions_only'}
                onChange={(e) => setFormat(e.target.value as any)}
              />
              <div className="radio-content">
                <strong>Action Logs Only (.json)</strong>
                <p>Frame-by-frame action annotations</p>
              </div>
            </label>
          </div>
        </section>

        <section className="export-section">
          <h2>Output Structure</h2>
          <div className="radio-group">
            <label className="radio-label">
              <input
                type="radio"
                name="structure"
                value="flat"
                checked={structure === 'flat'}
                onChange={(e) => setStructure(e.target.value as any)}
              />
              <div className="radio-content">
                <strong>Flat</strong>
                <p>All files in one folder</p>
              </div>
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name="structure"
                value="organized"
                checked={structure === 'organized'}
                onChange={(e) => setStructure(e.target.value as any)}
              />
              <div className="radio-content">
                <strong>Organized</strong>
                <p>Branch subfolders with categorized files</p>
              </div>
            </label>
          </div>
        </section>

        {format === 'training_package' && (
          <section className="export-section">
            <h2>Include</h2>
            <div className="checkbox-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={include.actionAnnotations}
                  onChange={(e) => setInclude({ ...include, actionAnnotations: e.target.checked })}
                />
                <span>Action annotations (frame-by-frame action logs)</span>
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={include.keyframeImages}
                  onChange={(e) => setInclude({ ...include, keyframeImages: e.target.checked })}
                />
                <span>Keyframe reference images</span>
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={include.promptTexts}
                  onChange={(e) => setInclude({ ...include, promptTexts: e.target.checked })}
                />
                <span>Prompt text files</span>
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={include.metadata}
                  onChange={(e) => setInclude({ ...include, metadata: e.target.checked })}
                />
                <span>Project metadata (action space, branch info)</span>
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={include.generationSettings}
                  onChange={(e) => setInclude({ ...include, generationSettings: e.target.checked })}
                />
                <span>Generation settings</span>
              </label>
            </div>
          </section>
        )}

        <div className="export-summary">
          <p>Total export size: ~{totalSize} MB</p>
        </div>

        <div className="export-actions">
          <Button onClick={handleExport} disabled={selectedBranches.length === 0}>
            Export to Folder
          </Button>
        </div>
      </div>
    </div>
  );
};
