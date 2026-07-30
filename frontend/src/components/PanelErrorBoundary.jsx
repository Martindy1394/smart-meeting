import { Component } from "react";

/**
 * Catches render/lifecycle errors in the workspace tree so a single panel
 * failure does not blank the whole app into an empty list-looking state.
 */
export default class PanelErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Keep console signal for ops; UI shows the user-facing panel below.
    console.error("PanelErrorBoundary caught", error, info?.componentStack);
  }

  render() {
    const { error } = this.state;
    if (error) {
      const title =
        this.props.title || "Something went wrong loading this panel";
      const detail = error?.message || String(error);
      return (
        <div className="panel-error-boundary" role="alert">
          <h3>{title}</h3>
          <p>{detail}</p>
          {typeof this.props.onRetry === "function" && (
            <button
              type="button"
              className="btn"
              onClick={() => {
                this.setState({ error: null });
                this.props.onRetry();
              }}
            >
              Try again
            </button>
          )}
        </div>
      );
    }
    return this.props.children;
  }
}
