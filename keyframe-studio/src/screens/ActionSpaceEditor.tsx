import React, { useState } from 'react';
import { useStore } from '../services/store';
import { Button } from '../components/Button';
import { Modal } from '../components/Modal';
import { validateActionName } from '../utils/helpers';
import './ActionSpaceEditor.css';

export const ActionSpaceEditor: React.FC = () => {
  const {
    currentProject,
    addCoreAction,
    editCoreAction,
    deleteCoreAction,
    addActionCombination,
    deleteActionCombination,
    setCurrentScreen
  } = useStore();

  const [newAction, setNewAction] = useState('');
  const [editingAction, setEditingAction] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [showCombinationModal, setShowCombinationModal] = useState(false);
  const [selectedActions, setSelectedActions] = useState<string[]>(['', '']);
  const [actionError, setActionError] = useState('');

  if (!currentProject) return null;

  const handleAddAction = () => {
    setActionError('');

    if (!newAction.trim()) {
      setActionError('Action name is required');
      return;
    }

    if (!validateActionName(newAction)) {
      setActionError('Action name must be in ALL_CAPS_SNAKE_CASE format');
      return;
    }

    if (currentProject.actionSpace.coreActions.includes(newAction)) {
      setActionError('Action already exists');
      return;
    }

    addCoreAction(newAction);
    setNewAction('');
  };

  const handleEditAction = () => {
    if (!editingAction || !editValue) return;

    setActionError('');

    if (!validateActionName(editValue)) {
      setActionError('Action name must be in ALL_CAPS_SNAKE_CASE format');
      return;
    }

    if (editValue !== editingAction && currentProject.actionSpace.coreActions.includes(editValue)) {
      setActionError('Action already exists');
      return;
    }

    editCoreAction(editingAction, editValue);
    setEditingAction(null);
    setEditValue('');
  };

  const handleAddCombination = () => {
    const filtered = selectedActions.filter(a => a.trim() !== '');
    if (filtered.length < 2) {
      setActionError('Select at least 2 actions');
      return;
    }

    // Check for duplicates in selection
    if (new Set(filtered).size !== filtered.length) {
      setActionError('Cannot select the same action twice');
      return;
    }

    const combination = filtered.join(' + ');

    if (currentProject.actionSpace.combinations.includes(combination)) {
      setActionError('Combination already exists');
      return;
    }

    addActionCombination(combination);
    setShowCombinationModal(false);
    setSelectedActions(['', '']);
    setActionError('');
  };

  const addActionSlot = () => {
    if (selectedActions.length < 5) {
      setSelectedActions([...selectedActions, '']);
    }
  };

  const updateActionSlot = (index: number, value: string) => {
    const updated = [...selectedActions];
    updated[index] = value;
    setSelectedActions(updated);
  };

  const previewCombination = selectedActions.filter(a => a.trim() !== '').join(' + ');

  const hasValidationErrors = () => {
    // Check all core actions have unique names
    const actions = currentProject.actionSpace.coreActions;
    if (new Set(actions).size !== actions.length) return true;

    // Check all combinations use only defined actions
    for (const combo of currentProject.actionSpace.combinations) {
      const parts = combo.split(' + ');
      if (parts.some(p => !actions.includes(p))) return true;
    }

    return false;
  };

  const validationErrors = [];
  const actions = currentProject.actionSpace.coreActions;
  if (new Set(actions).size !== actions.length) {
    validationErrors.push('Some core actions have duplicate names');
  }

  for (const combo of currentProject.actionSpace.combinations) {
    const parts = combo.split(' + ');
    const undefinedActions = parts.filter(p => !actions.includes(p));
    if (undefinedActions.length > 0) {
      validationErrors.push(`Combination "${combo}" uses undefined actions: ${undefinedActions.join(', ')}`);
    }
  }

  return (
    <div className="action-space-editor">
      <header className="editor-header">
        <div className="editor-header-left">
          <button className="back-btn" onClick={() => setCurrentScreen('canvas')}>
            ← Back to Project
          </button>
          <h1>{currentProject.name} &gt; Action Space</h1>
        </div>
        <Button
          onClick={() => setCurrentScreen('canvas')}
          disabled={hasValidationErrors()}
        >
          Save
        </Button>
      </header>

      <div className="editor-content">
        <p className="editor-description">
          Define all possible player actions in this game.
        </p>

        <div className="editor-columns">
          <div className="editor-column">
            <h2>Core Actions</h2>
            <div className="add-action-form">
              <input
                type="text"
                value={newAction}
                onChange={(e) => {
                  setNewAction(e.target.value.toUpperCase());
                  setActionError('');
                }}
                placeholder="ACTION_NAME"
                onKeyPress={(e) => e.key === 'Enter' && handleAddAction()}
              />
              <Button size="sm" onClick={handleAddAction}>
                + Add Action
              </Button>
            </div>
            {actionError && editingAction === null && newAction && (
              <span className="error-text">{actionError}</span>
            )}

            <div className="actions-list">
              {currentProject.actionSpace.coreActions.map((action) => (
                <div key={action} className="action-card">
                  {editingAction === action ? (
                    <div className="action-edit">
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => {
                          setEditValue(e.target.value.toUpperCase());
                          setActionError('');
                        }}
                        onKeyPress={(e) => e.key === 'Enter' && handleEditAction()}
                        autoFocus
                      />
                      <div className="action-card-actions">
                        <Button size="sm" onClick={handleEditAction}>
                          Save
                        </Button>
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => {
                            setEditingAction(null);
                            setEditValue('');
                            setActionError('');
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
                      {actionError && <span className="error-text">{actionError}</span>}
                    </div>
                  ) : (
                    <>
                      <span className="action-name">{action}</span>
                      <div className="action-card-actions">
                        <button
                          className="text-btn"
                          onClick={() => {
                            setEditingAction(action);
                            setEditValue(action);
                          }}
                        >
                          Edit
                        </button>
                        <button className="text-btn text-danger" onClick={() => deleteCoreAction(action)}>
                          Delete
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="editor-column">
            <h2>Valid Combinations</h2>
            <Button size="sm" onClick={() => setShowCombinationModal(true)}>
              + Add Combination
            </Button>

            <div className="actions-list">
              {currentProject.actionSpace.combinations.map((combination) => (
                <div key={combination} className="action-card">
                  <span className="action-name">{combination}</span>
                  <div className="action-card-actions">
                    <button
                      className="text-btn text-danger"
                      onClick={() => deleteActionCombination(combination)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="validation-section">
          <h3>Validation:</h3>
          {validationErrors.length === 0 ? (
            <>
              <div className="validation-item validation-success">
                ✓ All core actions have unique names
              </div>
              <div className="validation-item validation-success">
                ✓ All combinations use only defined core actions
              </div>
            </>
          ) : (
            validationErrors.map((error, index) => (
              <div key={index} className="validation-item validation-error">
                ✗ {error}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Add Combination Modal */}
      <Modal
        isOpen={showCombinationModal}
        onClose={() => {
          setShowCombinationModal(false);
          setSelectedActions(['', '']);
          setActionError('');
        }}
        title="Add Action Combination"
        footer={
          <>
            <Button
              variant="secondary"
              onClick={() => {
                setShowCombinationModal(false);
                setSelectedActions(['', '']);
                setActionError('');
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleAddCombination}>Add</Button>
          </>
        }
      >
        <div className="combination-form">
          <label>Select actions to combine:</label>
          {selectedActions.map((action, index) => (
            <select
              key={index}
              value={action}
              onChange={(e) => updateActionSlot(index, e.target.value)}
            >
              <option value="">Select action...</option>
              {currentProject.actionSpace.coreActions.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          ))}
          {selectedActions.length < 5 && (
            <Button size="sm" variant="secondary" onClick={addActionSlot}>
              + Add another action
            </Button>
          )}

          {previewCombination && (
            <div className="combination-preview">
              <strong>Preview:</strong> {previewCombination}
            </div>
          )}

          {actionError && <span className="error-text">{actionError}</span>}
        </div>
      </Modal>
    </div>
  );
};
