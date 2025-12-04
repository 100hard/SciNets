import { motion } from "framer-motion";
import { AgentNode } from "./AgentNode";

export const AgentsSection = () => {
  const agents = [
    {
      type: "planner" as const,
      title: "Planner",
      description:
        "Formulates research strategies and identifies key investigation paths.",
    },
    {
      type: "orchestrator" as const,
      title: "Orchestrator",
      description:
        "Coordinates information flow and manages task priorities.",
    },
    {
      type: "scientist" as const,
      title: "Scientist",
      description:
        "Explores literature, identifies patterns, generates hypotheses.",
    },
    {
      type: "critique" as const,
      title: "Critic",
      description:
        "Evaluates findings for validity and ensures scientific rigor.",
    },
  ];

  return (
    <section id="agents" className="relative py-24 px-6">
      <div className="max-w-2xl mx-auto">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-medium text-foreground mb-3">
            Four agents, one goal
          </h2>
          <p className="text-sm text-foreground-muted">
            Specialized AI working together to accelerate discovery.
          </p>
        </motion.div>

        {/* Agents list */}
        <div className="space-y-8">
          {agents.map((agent, index) => (
            <AgentNode
              key={agent.type}
              type={agent.type}
              title={agent.title}
              description={agent.description}
              delay={index}
            />
          ))}
        </div>
      </div>
    </section>
  );
};
