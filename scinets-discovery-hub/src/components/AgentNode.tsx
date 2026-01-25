import { motion } from "framer-motion";
import { cn } from "../lib/utils.ts";

type AgentType = "planner" | "orchestrator" | "scientist" | "critique" | "experiment";

interface AgentNodeProps {
  type: AgentType;
  title: string;
  description: string;
  delay?: number;
}

const agentConfig = {
  planner: {
    colorClass: "bg-agent-planner",
  },
  orchestrator: {
    colorClass: "bg-agent-orchestrator",
  },
  scientist: {
    colorClass: "bg-agent-scientist",
  },
  critique: {
    colorClass: "bg-agent-critique",
  },
  experiment: {
    colorClass: "bg-agent-experiment",
  },
};

export const AgentNode = ({
  type,
  title,
  description,
  delay = 0,
}: AgentNodeProps) => {
  const config = agentConfig[type];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: delay * 0.1 }}
      className="relative group"
    >
      {/* Connection point */}
      <div
        className={cn(
          "absolute -left-3 top-4 w-1.5 h-1.5 rounded-full",
          config.colorClass
        )}
      />

      <div className="pl-4 border-l border-border">
        <h3 className="text-sm font-medium text-foreground mb-1">{title}</h3>
        <p className="text-sm text-foreground-muted leading-relaxed">
          {description}
        </p>
      </div>
    </motion.div>
  );
};
