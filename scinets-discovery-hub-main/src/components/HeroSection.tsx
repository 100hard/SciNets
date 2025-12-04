import { motion } from "framer-motion";
import { ResearchInput } from "./ResearchInput";

export const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-16">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center max-w-2xl mx-auto mb-10"
      >
        {/* Main headline */}
        <h1 className="text-2xl md:text-3xl font-medium text-foreground leading-snug mb-6 tracking-tight">
          Human cognition has become the bottleneck of scientific progress.
          <span className="block text-foreground-muted mt-2">We're rewriting that constraint.</span>
        </h1>

        {/* Description */}
        <p className="text-sm text-foreground-muted max-w-xl mx-auto leading-relaxed">
          Operating over a continuously evolving scientific knowledge graph—mapping relationships across papers, entities, mechanisms, and evidence.
        </p>
      </motion.div>

      {/* Research input */}
      <ResearchInput onSubmit={(query) => console.log("Research:", query)} />

      {/* Minimal stats */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4, duration: 0.5 }}
        className="mt-16 flex items-center gap-12"
      >
        {[
          { value: "2.4M", label: "Papers" },
          { value: "156K", label: "Discoveries" },
          { value: "4", label: "Agents" },
        ].map((stat) => (
          <div key={stat.label} className="text-center">
            <div className="text-lg font-medium text-foreground">{stat.value}</div>
            <div className="text-xs text-foreground-muted">{stat.label}</div>
          </div>
        ))}
      </motion.div>
    </section>
  );
};
