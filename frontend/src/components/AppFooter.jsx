const YEAR = new Date().getFullYear();

export default function AppFooter({ className = "" }) {
  return (
    <footer className={`app-footer ${className}`.trim()}>
      <p className="app-footer-line">
        <span className="app-footer-copy">
          © {YEAR} Smart Meeting. All rights reserved.
        </span>
      </p>
    </footer>
  );
}
