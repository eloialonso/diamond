import React, { useState } from 'react';
import { useStore } from '../services/store';
import { Button } from '../components/Button';
import { readFileAsDataURL, getImageDimensions, calculateCost } from '../utils/helpers';
import './KeyframeCanvas.css';

export const KeyframeCanvas: React.FC = () => {
  const {
    currentProject,
    currentBranch,
    selectedKeyframe,
    selectedTransition,
    selectKeyframe,
    selectTransition,
    addKeyframe,
    updateKeyframe,
    deleteKeyframe,
    updateTransition,
    setCurrentScreen,
    switchBranch,
    createBranch,
    addToast
  } = useStore();

  const [newBranchName, setNewBranchName] = useState('');
  const [showBranchInput, setShowBranchInput] = useState(false);

  if (!currentProject || !currentBranch) return null;

  const sortedKeyframes = [...currentBranch.keyframes].sort((a, b) => a.timestamp - b.timestamp);

  const handleImageUpload = async (keyframeId: string, file: File) => {
    try {
      const dataUrl = await readFileAsDataURL(file);
      const dimensions = await getImageDimensions(dataUrl);

      updateKeyframe(keyframeId, {
        image: {
          path: file.name,
          dataUrl,
          width: dimensions.width,
          height: dimensions.height
        }
      });

      addToast({
        type: 'success',
        message: 'Image uploaded successfully',
        duration: 2000
      });
    } catch (error) {
      addToast({
        type: 'error',
        message: 'Failed to upload image',
        duration: 3000
      });
    }
  };

  const handleAddKeyframe = () => {
    const lastKeyframe = sortedKeyframes[sortedKeyframes.length - 1];
    const newTimestamp = lastKeyframe ? lastKeyframe.timestamp + 10 : 0;
    addKeyframe(newTimestamp, '', '');
    addToast({
      type: 'success',
      message: 'Keyframe added',
      duration: 2000
    });
  };

  const handleDeleteKeyframe = (keyframeId: string) => {
    if (window.confirm('Are you sure you want to delete this keyframe?')) {
      deleteKeyframe(keyframeId);
      addToast({
        type: 'success',
        message: 'Keyframe deleted',
        duration: 2000
      });
    }
  };

  const handleAddBranch = () => {
    if (!newBranchName.trim()) return;
    createBranch(newBranchName);
    setShowBranchInput(false);
    setNewBranchName('');
    addToast({
      type: 'success',
      message: 'Branch created',
      duration: 2000
    });
  };

  const getTransition = (sourceId: string, destId: string) => {
    return currentBranch.transitions.find(
      t => t.sourceKeyframe === sourceId && t.destKeyframe === destId
    );
  };

  const totalDuration = currentBranch.transitions.reduce((sum, t) => sum + t.duration, 0);

  return (
    <div className="keyframe-canvas">
      <header className="canvas-header">
        <div className="canvas-header-left">
          <button className="back-btn" onClick={() => setCurrentScreen('dashboard')}>
            ← Dashboard
          </button>
          <h1>{currentProject.name}</h1>
        </div>
        <div className="canvas-header-actions">
          <Button variant="secondary" onClick={() => setCurrentScreen('action-space')}>
            Edit Actions
          </Button>
          <Button variant="secondary" onClick={() => setCurrentScreen('generation')}>
            Generate
          </Button>
          <Button onClick={() => setCurrentScreen('export')}>Export</Button>
        </div>
      </header>

      <div className="canvas-toolbar">
        <div className="branch-selector">
          <label>Branches:</label>
          <select
            value={currentBranch.id}
            onChange={(e) => switchBranch(e.target.value)}
          >
            {currentProject.branches.map((branch) => (
              <option key={branch.id} value={branch.id}>
                {branch.name}
              </option>
            ))}
          </select>
          {showBranchInput ? (
            <div className="branch-input">
              <input
                type="text"
                value={newBranchName}
                onChange={(e) => setNewBranchName(e.target.value)}
                placeholder="Branch name..."
                onKeyPress={(e) => e.key === 'Enter' && handleAddBranch()}
                autoFocus
              />
              <Button size="sm" onClick={handleAddBranch}>
                Add
              </Button>
              <Button size="sm" variant="secondary" onClick={() => {
                setShowBranchInput(false);
                setNewBranchName('');
              }}>
                Cancel
              </Button>
            </div>
          ) : (
            <Button size="sm" onClick={() => setShowBranchInput(true)}>
              + Add Branch
            </Button>
          )}
        </div>
      </div>

      <div className="canvas-timeline">
        <h3>TIMELINE ({totalDuration} seconds)</h3>
        <div className="timeline">
          {sortedKeyframes.map((kf, index) => {
            const nextKf = sortedKeyframes[index + 1];
            const transition = nextKf ? getTransition(kf.id, nextKf.id) : null;

            return (
              <React.Fragment key={kf.id}>
                <div
                  className={`keyframe-node ${selectedKeyframe?.id === kf.id ? 'selected' : ''}`}
                  onClick={() => selectKeyframe(kf)}
                >
                  <div className="keyframe-label">KF{index + 1}</div>
                  <div className="keyframe-time">{kf.timestamp}s</div>
                </div>

                {transition && nextKf && (
                  <div
                    className={`transition-line ${
                      selectedTransition?.source.id === kf.id &&
                      selectedTransition?.dest.id === nextKf.id
                        ? 'selected'
                        : ''
                    }`}
                    onClick={() => selectTransition(kf, nextKf)}
                  >
                    <div className="transition-duration">{transition.duration}s</div>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      <div className="canvas-panels">
        {selectedKeyframe && (
          <div className="panel keyframe-panel">
            <h3>Keyframe {sortedKeyframes.findIndex(k => k.id === selectedKeyframe.id) + 1}</h3>

            <div className="image-preview">
              {selectedKeyframe.image.dataUrl ? (
                <img src={selectedKeyframe.image.dataUrl} alt="Keyframe" />
              ) : (
                <div className="image-placeholder">
                  <p>No image uploaded</p>
                </div>
              )}
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleImageUpload(selectedKeyframe.id, file);
                }}
                id="image-upload"
                style={{ display: 'none' }}
              />
              <label htmlFor="image-upload" className="upload-btn">
                Click to upload/change
              </label>
            </div>

            <div className="form-group">
              <label>Timestamp (seconds):</label>
              <input
                type="number"
                value={selectedKeyframe.timestamp}
                onChange={(e) =>
                  updateKeyframe(selectedKeyframe.id, { timestamp: Number(e.target.value) })
                }
                min="0"
              />
            </div>

            <div className="form-group">
              <label>Action at this keyframe:</label>
              <select
                value={selectedKeyframe.action}
                onChange={(e) => updateKeyframe(selectedKeyframe.id, { action: e.target.value })}
              >
                <option value="">(No action)</option>
                {currentProject.actionSpace.coreActions.map((action) => (
                  <option key={action} value={action}>
                    {action}
                  </option>
                ))}
                {currentProject.actionSpace.combinations.map((combo) => (
                  <option key={combo} value={combo}>
                    {combo}
                  </option>
                ))}
              </select>
            </div>

            <Button variant="danger" onClick={() => handleDeleteKeyframe(selectedKeyframe.id)}>
              Delete Keyframe
            </Button>
          </div>
        )}

        {selectedTransition && (() => {
          const transition = getTransition(selectedTransition.source.id, selectedTransition.dest.id);
          if (!transition) return null;

          return (
            <div className="panel prompt-panel">
              <h3>
                KF{sortedKeyframes.findIndex(k => k.id === selectedTransition.source.id) + 1} →{' '}
                KF{sortedKeyframes.findIndex(k => k.id === selectedTransition.dest.id) + 1}
              </h3>

              <div className="form-group">
                <label>Prompt:</label>
                <textarea
                  value={transition.prompt}
                  onChange={(e) =>
                    updateTransition(selectedTransition.source.id, selectedTransition.dest.id, {
                      prompt: e.target.value
                    })
                  }
                  placeholder="Describe what happens between these keyframes..."
                  rows={10}
                  maxLength={2000}
                />
                <span className="char-count">
                  {transition.prompt.length} / 2000 characters
                </span>
              </div>

              <div className="form-group">
                <label>Action:</label>
                <select
                  value={transition.action}
                  onChange={(e) =>
                    updateTransition(selectedTransition.source.id, selectedTransition.dest.id, {
                      action: e.target.value
                    })
                  }
                >
                  <option value="">(No action)</option>
                  {currentProject.actionSpace.coreActions.map((action) => (
                    <option key={action} value={action}>
                      {action}
                    </option>
                  ))}
                </select>
              </div>

              <div className="info-group">
                <p>
                  <strong>Duration:</strong> {transition.duration} seconds
                </p>
                <p>
                  <strong>Cost estimate:</strong> $
                  {calculateCost(transition.duration, currentProject.settings.defaultModel).toFixed(2)}
                </p>
              </div>
            </div>
          );
        })()}
      </div>

      <div className="canvas-actions">
        <Button onClick={handleAddKeyframe}>+ Add Keyframe After</Button>
      </div>
    </div>
  );
};
