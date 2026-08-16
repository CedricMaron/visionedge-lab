import { useEffect, useState } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { Icon } from './components/Icon';
import { api } from './services/api';
import { useClassStore } from './stores/classStore';

import PlaygroundPage from './pages/PlaygroundPage';
import PipelinePage from './pages/PipelinePage';
import PerformancePage from './pages/PerformancePage';
import ModelsPage from './pages/ModelsPage';
import EnvironmentPage from './pages/EnvironmentPage';
import SettingsPage from './pages/SettingsPage';
import ArchitecturePage from './pages/ArchitecturePage';
import ResearchNotesPage from './pages/ResearchNotesPage';
import RunDetailPage from './pages/lab/RunDetailPage';

// Populate the persisted class catalog once on startup.
function useCatalogLoader() {
  const setCatalog = useClassStore((s) => s.setCatalog);
  useEffect(() => {
    const controller = new AbortController();
    api
      .classes(controller.signal)
      .then((res) => setCatalog(res.classes, res.groups))
      .catch(() => {
        /* Offline / backend down — pages surface their own errors. */
      });
    return () => controller.abort();
  }, [setCatalog]);
}

export default function App() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const location = useLocation();
  useCatalogLoader();

  useEffect(() => {
    setMobileNavOpen(false);
  }, [location.pathname]);

  return (
    <div className="flex h-screen overflow-hidden bg-canvas">
      {/* Desktop sidebar */}
      <aside className="hidden w-60 shrink-0 border-r border-subtle lg:block">
        <Sidebar />
      </aside>

      {/* Mobile drawer */}
      {mobileNavOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setMobileNavOpen(false)}
            aria-hidden="true"
          />
          <div className="absolute left-0 top-0 h-full w-72 max-w-[85vw] border-r border-subtle shadow-xl">
            <button
              className="btn-ghost absolute right-2 top-2 z-10 px-2 py-2"
              onClick={() => setMobileNavOpen(false)}
              aria-label="Close navigation"
            >
              <Icon name="close" className="h-5 w-5" />
            </button>
            <Sidebar onNavigate={() => setMobileNavOpen(false)} />
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onMenu={() => setMobileNavOpen(true)} />
        <main className="flex-1 overflow-y-auto px-3 py-4 sm:px-6 sm:py-5 lg:px-8">
          <div className="mx-auto max-w-6xl pb-16">
            {/* Keyed on the path so navigating away from a crashed page clears it. */}
            <ErrorBoundary resetKey={location.pathname}>
              <Routes>
                <Route path="/" element={<PlaygroundPage />} />
                <Route path="/pipeline" element={<PipelinePage />} />
                <Route path="/performance" element={<PerformancePage />} />
                <Route path="/models" element={<ModelsPage />} />
                <Route path="/environment" element={<EnvironmentPage />} />
                <Route path="/runs/:runId" element={<RunDetailPage />} />

                <Route path="/settings" element={<SettingsPage />} />
                <Route path="/about" element={<ArchitecturePage />} />
                <Route path="/about/research" element={<ResearchNotesPage />} />

                {/* Pre-consolidation URLs, kept so existing links and bookmarks land
                  on the page that absorbed the functionality. */}
                <Route path="/live" element={<Navigate to="/" replace />} />
                <Route path="/assistant" element={<Navigate to="/" replace />} />
                <Route path="/classes" element={<Navigate to="/" replace />} />
                <Route path="/capabilities" element={<Navigate to="/environment" replace />} />
                <Route path="/lab/system" element={<Navigate to="/environment" replace />} />
                <Route path="/lab/models" element={<Navigate to="/models" replace />} />
                <Route path="/lab/run" element={<Navigate to="/performance" replace />} />
                <Route path="/lab/results" element={<Navigate to="/performance" replace />} />
                <Route path="/lab/results/:runId" element={<LegacyRunRedirect />} />
                <Route path="/benchmarks" element={<Navigate to="/performance" replace />} />
                <Route path="/architecture" element={<Navigate to="/about" replace />} />
                <Route path="/research" element={<Navigate to="/about/research" replace />} />

                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </ErrorBoundary>
          </div>
        </main>
      </div>
    </div>
  );
}

/** /lab/results/:runId kept its run id; only the prefix changed. */
function LegacyRunRedirect() {
  const location = useLocation();
  const runId = location.pathname.split('/').pop() ?? '';
  return <Navigate to={`/runs/${runId}`} replace />;
}
