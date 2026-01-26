import { motion } from "framer-motion";

const features = [
  {
    title: "Dynamic Knowledge Graph",
    description: "Automatically extract concepts from papers and build an evolving graph of scientific relationships.",
  },
  {
    title: "Multi-hop Reasoning",
    description: "Traverse reasoning paths across the graph to discover non-obvious connections and generate testable hypotheses.",
  },
  {
    title: "Evidence Scoring",
    description: "Gather supporting, contradicting, and neutral evidence with confidence scores for each hypothesis.",
  },
  {
    title: "Self-correcting Experiments",
    description: "Write and execute Python experiments with automatic error correction and result validation.",
  },
];

export const FeaturesSection = () => {
  return (
    <section className="relative py-24 px-6 border-t border-border">
      <div className="max-w-2xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-medium text-foreground mb-3">
            Capabilities
          </h2>
          <p className="text-sm text-foreground-muted">
            Tools designed to accelerate scientific discovery.
          </p>
        </motion.div>

        <div className="grid sm:grid-cols-2 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.05 }}
            >
              <h3 className="text-sm font-medium text-foreground mb-1">
                {feature.title}
              </h3>
              <p className="text-sm text-foreground-muted leading-relaxed text-justify">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
