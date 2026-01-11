import { motion } from "framer-motion";
import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";

export const Header = () => {
  const location = useLocation();
  const isDiscovery = location.pathname === "/discovery";

  return (
    <motion.header
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-sm border-b border-border/50"
    >
      <div className="mx-auto max-w-5xl px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-foreground" />
            <span className="text-sm font-medium tracking-tight text-foreground">
              SciNets
            </span>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <Link
              to="/"
              className={cn(
                "text-sm transition-colors",
                location.pathname === "/" ? "text-foreground" : "text-foreground-muted hover:text-foreground"
              )}
            >
              Home
            </Link>
            <Link
              to="/discovery"
              className={cn(
                "text-sm transition-colors",
                isDiscovery ? "text-foreground" : "text-foreground-muted hover:text-foreground"
              )}
            >
              Discovery
            </Link>
            <a
              href="#agents"
              className="text-sm text-foreground-muted hover:text-foreground transition-colors"
            >
              Agents
            </a>
          </nav>

          {/* CTA */}
          <Link
            to="/login"
            className={cn(
              "px-3 py-1.5 rounded text-sm",
              "bg-foreground text-background",
              "hover:bg-foreground/90 transition-colors"
            )}
          >
            Login
          </Link>
        </div>
      </div>
    </motion.header>
  );
};
