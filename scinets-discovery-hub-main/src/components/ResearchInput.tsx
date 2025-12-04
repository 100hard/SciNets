import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface ResearchInputProps {
  onSubmit?: (query: string) => void;
}

export const ResearchInput = ({ onSubmit }: ResearchInputProps) => {
  const [query, setQuery] = useState("");
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && onSubmit) {
      onSubmit(query);
    }
  };

  return (
    <motion.form
      onSubmit={handleSubmit}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="w-full max-w-lg mx-auto"
    >
      <div
        className={cn(
          "relative rounded-lg overflow-hidden",
          "border transition-colors duration-200",
          isFocused ? "border-foreground/20" : "border-border"
        )}
      >
        <div className="flex items-center gap-3 px-4 py-3 bg-background-elevated">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Describe your research question..."
            className={cn(
              "flex-1 bg-transparent",
              "text-foreground placeholder:text-foreground-muted",
              "text-sm outline-none"
            )}
          />

          <button
            type="submit"
            className={cn(
              "p-2 rounded",
              "bg-foreground text-background",
              "hover:bg-foreground/90 transition-colors",
              "disabled:opacity-30"
            )}
            disabled={!query.trim()}
          >
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Suggestion chips */}
      <div className="mt-3 flex flex-wrap justify-center gap-2">
        {[
          "CRISPR advances",
          "Quantum algorithms",
          "Climate modeling",
        ].map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => setQuery(suggestion)}
            className={cn(
              "px-2.5 py-1 rounded",
              "text-xs text-foreground-muted",
              "border border-border",
              "hover:border-foreground/20 hover:text-foreground",
              "transition-colors"
            )}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </motion.form>
  );
};
