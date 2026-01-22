import { useState } from "react";
import { motion } from "framer-motion";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ContactModal } from "./ContactModal";

export const Header = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [showContact, setShowContact] = useState(false);
  const isDiscovery = location.pathname === "/discovery";

  const handleHomeClick = (e: React.MouseEvent) => {
    if (location.pathname === "/") {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const handleAgentsClick = (e: React.MouseEvent) => {
    e.preventDefault();
    if (location.pathname !== "/") {
      navigate("/#agents");
    } else {
      const element = document.getElementById("agents");
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }
  };

  return (
    <>
      <motion.header
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-sm border-b border-border/50"
      >
        <div className="w-full px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2" onClick={handleHomeClick}>
              <div className="w-2 h-2 rounded-full bg-foreground" />
              <span className="text-sm font-medium tracking-tight text-foreground">
                SciNets
              </span>
            </Link>

            {/* Navigation */}
            <nav className="hidden md:flex items-center gap-8">
              <Link
                to="/"
                onClick={handleHomeClick}
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
                onClick={handleAgentsClick}
                className="text-sm text-foreground-muted hover:text-foreground transition-colors cursor-pointer"
              >
                Agents
              </a>
              <button
                onClick={() => setShowContact(true)}
                className="text-sm text-foreground-muted hover:text-foreground transition-colors"
              >
                Contact
              </button>
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
      <ContactModal isOpen={showContact} onClose={() => setShowContact(false)} />
    </>
  );
};
