import { motion } from "framer-motion";

const features = [
  {
    title: "Literature Synthesis",
    description: "Synthesize literature at scale, mapping relationships across papers, entities, and mechanisms.",
  },
  {
    title: "Hypothesis Generation",
    description: "Generate structured and scientifically grounded hypotheses from identified patterns.",
  },
  {
    title: "Parallel Workflows",
    description: "Design and execute parallel experimental workflows across research domains.",
  },
  {
    title: "Iterative Reasoning",
    description: "Refine insights through iterative reasoning cycles and evidence integration.",
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
              <p className="text-sm text-foreground-muted leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
