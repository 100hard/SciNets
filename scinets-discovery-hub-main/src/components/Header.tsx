import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export const Header = () => {
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
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-foreground" />
            <span className="text-sm font-medium tracking-tight text-foreground">
              SciNets
            </span>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            {["Research", "Agents", "About"].map((item) => (
              <a
                key={item}
                href={`#${item.toLowerCase()}`}
                className="text-sm text-foreground-muted hover:text-foreground transition-colors"
              >
                {item}
              </a>
            ))}
          </nav>

          {/* CTA */}
          <button
            className={cn(
              "px-3 py-1.5 rounded text-sm",
              "bg-foreground text-background",
              "hover:bg-foreground/90 transition-colors"
            )}
          >
            Get Started
          </button>
        </div>
      </div>
    </motion.header>
  );
};
