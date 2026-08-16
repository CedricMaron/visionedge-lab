/**
 * Last line of defence between a component exception and a white screen.
 *
 * React unmounts the entire tree when a render or effect throws, which is exactly
 * what a visitor reports as "the page is blank" — with no clue as to which part
 * failed. Catching it here keeps the shell, the navigation and the error itself on
 * screen, so the next question is "why did this page fail" rather than "is the
 * server down".
 *
 * It cannot catch everything: an exception thrown before the first render, or
 * inside an async callback, never reaches a boundary. It covers the common case.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  /** Remounting on navigation: a boundary that has caught stays caught otherwise. */
  resetKey?: string;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(previous: Props) {
    if (this.state.error && previous.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Kept: the console trace is what makes this diagnosable from a bug report.
    console.error('page crashed', error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="card card-pad border-bad/30 bg-bad/5">
        <h1 className="text-base font-semibold text-bad">This page failed to render</h1>
        <p className="mt-1 text-sm text-secondary">
          The rest of the application still works — use the navigation to move elsewhere.
        </p>
        <pre className="mt-3 overflow-x-auto rounded border border-subtle bg-elevated p-3 font-mono text-2xs text-secondary">
          {error.message}
        </pre>
        <button className="btn-ghost mt-3" onClick={() => window.location.reload()}>
          Reload
        </button>
      </div>
    );
  }
}
