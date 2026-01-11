import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center max-w-2xl mx-auto"
      >
        {/* Main headline */}
        <h1 className="text-2xl md:text-3xl font-medium text-foreground leading-snug mb-6 tracking-tight">
          Human cognition has become the bottleneck of scientific progress.
          <span className="block text-foreground-muted mt-2">We're rewriting that constraint.</span>
        </h1>

        {/* Description */}
        <p className="text-sm text-foreground-muted max-w-xl mx-auto leading-relaxed mb-8">
          SciNets autonomously reads research papers, builds a dynamic knowledge graph, proposes hypotheses, gathers evidence, and runs self-correcting experiments.
        </p>

        {/* CTA Button */}
        <Link to="/discovery">
          <Button
            variant="outline"
            className="border-foreground/20 text-foreground hover:bg-foreground hover:text-background transition-colors"
          >
            Start a Discovery
          </Button>
        </Link>
      </motion.div>
    </section>
  );
};
