const YEAR = new Date().getFullYear();

export default function AppFooter({ className = "" }) {
  return (
    <footer className={`app-footer ${className}`.trim()}>
      <p className="app-footer-copy">
        © {YEAR} Smart Meeting. All rights reserved.
      </p>
      <p className="app-footer-property">
        A property of Iloilo Science and Technology
      </p>
    </footer>
  );
}
