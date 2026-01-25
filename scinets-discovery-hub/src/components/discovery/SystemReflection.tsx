import { motion } from "framer-motion";
import {
  AlertTriangle, CheckCircle2, HelpCircle,
  Compass, MessageSquare
} from "lucide-react";
import { cn } from "../../lib/utils";

interface ReflectionData {
  stronglySupported: string[];
  unresolvedBridges: string[];
  suggestedNextSteps: string[];
  overallConfidence: "high" | "moderate" | "low";
  caveats: string[];
}

interface SystemReflectionProps {
  reflection: ReflectionData;
}

export const SystemReflection = ({ reflection }: SystemReflectionProps) => {
  const confidenceConfig = {
    high: { label: "High confidence", color: "text-emerald-400" },
    moderate: { label: "Moderate confidence", color: "text-amber-400" },
    low: { label: "Low confidence", color: "text-red-400" },
  };

  const config = confidenceConfig[reflection.overallConfidence];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="border border-border rounded-lg bg-background/30 p-6"
    >
      <div className="flex items-center gap-2 mb-4">
        <MessageSquare className="w-4 h-4 text-foreground-muted" />
        <h3 className="text-sm font-medium text-foreground">System Assessment</h3>
        <span className={cn("text-xs ml-auto", config.color)}>
          {config.label}
        </span>
      </div>

      {/* Caveats / Uncertainty notice */}
      <div className="mb-6 p-3 border border-border rounded bg-foreground/5">
        <div className="flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs text-foreground-muted mb-2">
              This analysis is based on automated literature review and hypothesis generation.
              Results should be validated by domain experts before use in research decisions.
            </p>
            {reflection.caveats.length > 0 && (
              <ul className="text-xs text-foreground-muted space-y-1">
                {reflection.caveats.map((caveat, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-foreground-muted mx-2" />
                    {caveat}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <div className="space-y-5">
        {/* Strongly Supported Mechanisms */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <h4 className="text-xs font-medium text-foreground">Strongly Supported Mechanisms</h4>
          </div>
          <ul className="space-y-1.5 pl-5">
            {reflection.stronglySupported.map((item, i) => (
              <li key={i} className="text-xs text-foreground-muted leading-relaxed">
                {item}
              </li>
            ))}
          </ul>
        </div>

        {/* Unresolved or Failed Conceptual Bridges */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
            <h4 className="text-xs font-medium text-foreground">Unresolved Conceptual Bridges</h4>
          </div>
          <ul className="space-y-1.5 pl-5">
            {reflection.unresolvedBridges.map((item, i) => (
              <li key={i} className="text-xs text-foreground-muted leading-relaxed">
                {item}
              </li>
            ))}
          </ul>
        </div>

        {/* Suggested Next Steps */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Compass className="w-3.5 h-3.5 text-foreground-muted" />
            <h4 className="text-xs font-medium text-foreground">Suggested Next Exploration Steps</h4>
          </div>
          <ul className="space-y-1.5 pl-5">
            {reflection.suggestedNextSteps.map((step, i) => (
              <li key={i} className="text-xs text-foreground-muted leading-relaxed flex items-start gap-2">
                <span className="text-foreground/40">{i + 1}.</span>
                {step}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </motion.div>
  );
};

export type { ReflectionData };
