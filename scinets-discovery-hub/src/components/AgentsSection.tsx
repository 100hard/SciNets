import { motion } from "framer-motion";
import { AgentNode } from "./AgentNode";

export const AgentsSection = () => {
  const agents = [
    {
      type: "scientist" as const,
      title: "Literature Agent",
      description:
        "Reads papers at scale, extracts concepts, and builds a dynamic scientific knowledge graph.",
    },
    {
      type: "planner" as const,
      title: "Hypothesis Agent",
      description:
        "Traverses multi-hop reasoning paths on the graph to generate structured, testable hypotheses.",
    },
    {
      type: "orchestrator" as const,
      title: "Evidence Agent",
      description:
        "Searches the literature for supporting, contradicting, and neutral evidence with confidence scoring.",
    },
    {
      type: "experiment" as const,
      title: "Experiment Agent",
      description:
        "Writes and executes Python experiments using self-correcting loops to validate hypotheses.",
    },
    {
      type: "critique" as const,
      title: "Critic Agent",
      description:
        "Evaluates results, interprets plots, and produces a scientific-grade assessment of the findings.",
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
            Five agents, one discovery loop
          </h2>
          <p className="text-sm text-foreground-muted">
            AI specialists collaborating to read the literature, propose hypotheses, gather evidence, and run experiments.
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
