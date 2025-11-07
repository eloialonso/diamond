import React, { useState } from 'react';
import { useStore } from '../services/store';
import { Button } from '../components/Button';
import { Modal } from '../components/Modal';
import { formatDate } from '../utils/helpers';
import { Project } from '../types';
import './ProjectsDashboard.css';

export const ProjectsDashboard: React.FC = () => {
  const { projects, createProject, deleteProject, duplicateProject, loadProject } = useStore();
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null);
  const [newProjectName, setNewProjectName] = useState('');
  const [nameError, setNameError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState<'all' | 'in_progress' | 'completed' | 'not_started'>('all');

  const handleCreateProject = () => {
    // Validation
    if (!newProjectName.trim()) {
      setNameError('Project name is required');
      return;
    }

    if (projects.some(p => p.name === newProjectName)) {
      setNameError('Project name already exists');
      return;
    }

    createProject(newProjectName);
    setShowNewProjectModal(false);
    setNewProjectName('');
    setNameError('');
  };

  const handleDeleteProject = () => {
    if (projectToDelete) {
      deleteProject(projectToDelete.id);
      setShowDeleteModal(false);
      setProjectToDelete(null);
    }
  };

  const getProjectStatus = (project: Project): 'complete' | 'in_progress' | 'not_started' => {
    const allTransitions = project.branches.flatMap(b => b.transitions);
    if (allTransitions.length === 0) return 'not_started';

    const completed = allTransitions.filter(t => t.generationStatus === 'complete').length;
    if (completed === allTransitions.length) return 'complete';
    if (completed > 0) return 'in_progress';
    return 'not_started';
  };

  const filteredProjects = projects.filter(project => {
    const matchesSearch = project.name.toLowerCase().includes(searchQuery.toLowerCase());
    if (!matchesSearch) return false;

    if (filter === 'all') return true;
    return getProjectStatus(project) === filter;
  });

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Keyframe Studio</h1>
        <div className="dashboard-header-actions">
          <button className="icon-btn">Settings</button>
          <button className="icon-btn">Help</button>
        </div>
      </header>

      <div className="dashboard-content">
        <div className="dashboard-toolbar">
          <h2>Projects</h2>
          <Button onClick={() => setShowNewProjectModal(true)}>+ New Project</Button>
        </div>

        <div className="dashboard-filters">
          <select value={filter} onChange={(e) => setFilter(e.target.value as any)}>
            <option value="all">All Projects</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
            <option value="not_started">Not Started</option>
          </select>
          <input
            type="text"
            placeholder="Search projects..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>

        {filteredProjects.length === 0 ? (
          <div className="empty-state">
            <h3>{projects.length === 0 ? 'Welcome to Keyframe Studio' : 'No projects found'}</h3>
            <p>
              {projects.length === 0
                ? "You haven't created any projects yet."
                : 'Try adjusting your search or filter.'}
            </p>
            {projects.length === 0 && (
              <Button onClick={() => setShowNewProjectModal(true)}>
                Create Your First Project
              </Button>
            )}
          </div>
        ) : (
          <div className="projects-grid">
            {filteredProjects.map((project) => {
              const status = getProjectStatus(project);
              const totalKeyframes = project.branches.reduce((sum, b) => sum + b.keyframes.length, 0);
              const totalTransitions = project.branches.reduce((sum, b) => sum + b.transitions.length, 0);
              const totalDuration = project.branches.reduce((sum, b) =>
                sum + b.transitions.reduce((s, t) => s + t.duration, 0), 0
              );
              const completedTransitions = project.branches.flatMap(b => b.transitions)
                .filter(t => t.generationStatus === 'complete').length;

              return (
                <div key={project.id} className="project-card">
                  <div className="project-card-header">
                    <h3>{project.name}</h3>
                    <div className={`status-indicator status-${status}`}>
                      {status === 'complete' && '●'}
                      {status === 'in_progress' && '◐'}
                      {status === 'not_started' && '○'}
                    </div>
                  </div>

                  <div className="project-card-meta">
                    <span className="text-sm">Created: {formatDate(project.created)}</span>
                    <span className="text-sm">Modified: {formatDate(project.modified)}</span>
                  </div>

                  <div className="project-card-stats">
                    <span>{project.branches.length} branches</span>
                    <span>{totalKeyframes} keyframes</span>
                    <span>{totalDuration}s total</span>
                  </div>

                  <div className="project-card-status">
                    {status === 'complete' && (
                      <span className="status-text">✓ All generated</span>
                    )}
                    {status === 'in_progress' && (
                      <span className="status-text">
                        ◐ {completedTransitions} / {totalTransitions} generated
                      </span>
                    )}
                    {status === 'not_started' && (
                      <span className="status-text">○ Not started</span>
                    )}
                  </div>

                  <div className="project-card-actions">
                    <Button onClick={() => loadProject(project.id)}>Open</Button>
                    <Button variant="secondary" onClick={() => duplicateProject(project.id)}>
                      Duplicate
                    </Button>
                    <Button
                      variant="danger"
                      onClick={() => {
                        setProjectToDelete(project);
                        setShowDeleteModal(true);
                      }}
                    >
                      Delete
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* New Project Modal */}
      <Modal
        isOpen={showNewProjectModal}
        onClose={() => {
          setShowNewProjectModal(false);
          setNewProjectName('');
          setNameError('');
        }}
        title="Create New Project"
        footer={
          <>
            <Button
              variant="secondary"
              onClick={() => {
                setShowNewProjectModal(false);
                setNewProjectName('');
                setNameError('');
              }}
            >
              Cancel
            </Button>
            <Button onClick={handleCreateProject}>Create</Button>
          </>
        }
      >
        <div className="form-group">
          <label htmlFor="project-name">Project Name:</label>
          <input
            id="project-name"
            type="text"
            value={newProjectName}
            onChange={(e) => {
              setNewProjectName(e.target.value);
              setNameError('');
            }}
            placeholder="Enter project name..."
            maxLength={100}
            autoFocus
          />
          {nameError && <span className="error-text">{nameError}</span>}
        </div>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={showDeleteModal}
        onClose={() => {
          setShowDeleteModal(false);
          setProjectToDelete(null);
        }}
        title="Delete Project"
        footer={
          <>
            <Button
              variant="secondary"
              onClick={() => {
                setShowDeleteModal(false);
                setProjectToDelete(null);
              }}
            >
              Cancel
            </Button>
            <Button variant="danger" onClick={handleDeleteProject}>
              Delete
            </Button>
          </>
        }
      >
        <div className="delete-confirmation">
          <p>
            Are you sure you want to delete <strong>"{projectToDelete?.name}"</strong>?
          </p>
          <p className="mt-4">This will delete:</p>
          <ul>
            <li>All keyframes</li>
            <li>All branches</li>
            <li>All generated videos</li>
          </ul>
          <p className="warning-text mt-4">This action cannot be undone.</p>
        </div>
      </Modal>
    </div>
  );
};
