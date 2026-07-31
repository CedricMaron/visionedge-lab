import { useEffect, useState } from 'react';
import { Route, Routes, useLocation } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { Icon } from './components/Icon';
import { api } from './services/api';
import { useClassStore } from './stores/classStore';

import LiveInferencePage from './pages/LiveInferencePage';
import DeviceCapabilitiesPage from './pages/DeviceCapabilitiesPage';
import ModelSelectorPage from './pages/ModelSelectorPage';
import ClassSelectorPage from './pages/ClassSelectorPage';
import PerformancePage from './pages/PerformancePage';
import BenchmarksPage from './pages/BenchmarksPage';
import SettingsPage from './pages/SettingsPage';
import MultimodalAssistantPage from './pages/MultimodalAssistantPage';
import ArchitecturePage from './pages/ArchitecturePage';
import ResearchNotesPage from './pages/ResearchNotesPage';
import OverviewPage from './pages/lab/OverviewPage';
import RunBenchmarkPage from './pages/lab/RunBenchmarkPage';
import ResultsPage from './pages/lab/ResultsPage';
import RunDetailPage from './pages/lab/RunDetailPage';
import LabModelsPage from './pages/lab/LabModelsPage';
import SystemPage from './pages/lab/SystemPage';
import {
  TemporalSceneAnalysisPage,
  WorldModelLabPage,
  JepaTrainingPage,
  EmbeddingExplorerPage,
  AnomalyDetectionPage,
  CrossModalSearchPage,
  MultimodalBenchmarksPage,
  ServerConnectionsPage,
  LogsPage,
  ModelComparisonPage,
  OptimizationAdvisorPage,
} from './pages/PlannedPages';

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
      <aside className="hidden w-64 shrink-0 border-r border-subtle lg:block">
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
          <div className="absolute left-0 top-0 h-full w-72 border-r border-subtle shadow-xl">
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
        <main className="flex-1 overflow-y-auto px-4 py-5 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-6xl pb-16">
            <Routes>
              {/* InferenceLab: benchmarking is now the primary surface. */}
              <Route path="/" element={<OverviewPage />} />
              <Route path="/lab/run" element={<RunBenchmarkPage />} />
              <Route path="/lab/results" element={<ResultsPage />} />
              <Route path="/lab/results/:runId" element={<RunDetailPage />} />
              <Route path="/lab/models" element={<LabModelsPage />} />
              <Route path="/lab/system" element={<SystemPage />} />

              {/* Vision slice, preserved from VisionEdge Lab. */}
              <Route path="/live" element={<LiveInferencePage />} />
              <Route path="/models" element={<ModelSelectorPage />} />
              <Route path="/classes" element={<ClassSelectorPage />} />
              <Route path="/assistant" element={<MultimodalAssistantPage />} />
              <Route path="/capabilities" element={<DeviceCapabilitiesPage />} />
              <Route path="/performance" element={<PerformancePage />} />
              <Route path="/benchmarks" element={<BenchmarksPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/architecture" element={<ArchitecturePage />} />
              <Route path="/research" element={<ResearchNotesPage />} />

              <Route path="/temporal" element={<TemporalSceneAnalysisPage />} />
              <Route path="/world-model" element={<WorldModelLabPage />} />
              <Route path="/jepa" element={<JepaTrainingPage />} />
              <Route path="/embeddings" element={<EmbeddingExplorerPage />} />
              <Route path="/anomaly" element={<AnomalyDetectionPage />} />
              <Route path="/cross-modal" element={<CrossModalSearchPage />} />
              <Route path="/multimodal-benchmarks" element={<MultimodalBenchmarksPage />} />
              <Route path="/servers" element={<ServerConnectionsPage />} />
              <Route path="/logs" element={<LogsPage />} />
              <Route path="/model-comparison" element={<ModelComparisonPage />} />
              <Route path="/optimization" element={<OptimizationAdvisorPage />} />

              <Route path="*" element={<OverviewPage />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}
