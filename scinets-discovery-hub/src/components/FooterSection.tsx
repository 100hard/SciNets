export const FooterSection = () => {
  return (
    <footer className="relative py-12 px-6 border-t border-border">
      <div className="max-w-2xl mx-auto">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-foreground" />
            <span className="text-sm text-foreground">SciNets</span>
          </div>

          {/* Copyright */}
          <p className="text-xs text-foreground-muted">
            © 2026 SciNets
          </p>
        </div>
      </div>
    </footer>
  );
};
