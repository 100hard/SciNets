import { useState, useEffect } from "react";
import { Loader2, Terminal } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface SearchOverlayProps {
    query: string;
}

export const SearchOverlay = ({ query }: SearchOverlayProps) => {
    const [logs, setLogs] = useState<string[]>([]);

    const steps = [
        "Initializing Literature Agent...",
        `Parsing query semantics: "${query}"...`,
        "Connecting to OpenAlex Knowledge Graph...",
        "Retrieving candidate papers...",
        "Analyzing citation network density...",
        "Evaluating semantic relevance...",
        "Filtering low-confidence results...",
        "Structuring candidate corpus..."
    ];

    useEffect(() => {
        let stepIndex = 0;

        const interval = setInterval(() => {
            if (stepIndex < steps.length) {
                setLogs(prev => [...prev, steps[stepIndex]]);
                stepIndex++;
            } else {
                clearInterval(interval);
            }
        }, 800); // New log every 800ms

        return () => clearInterval(interval);
    }, [query]);

    return (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-background/95 backdrop-blur-sm">
            <div className="w-full max-w-md p-6 space-y-6">
                <div className="flex flex-col items-center text-center space-y-2">
                    <Loader2 className="w-8 h-8 text-agent-scientist animate-spin" />
                    <h3 className="text-xl font-medium text-foreground">Literature Agent Active</h3>
                    <p className="text-sm text-foreground-muted">Curating research corpus for your session</p>
                </div>

                <div className="bg-black/40 border border-border rounded-lg p-4 h-64 overflow-hidden flex flex-col font-mono text-xs relative">
                    <div className="flex items-center gap-2 text-foreground-muted mb-2 border-b border-border/50 pb-2">
                        <Terminal className="w-3 h-3" />
                        <span>agent_activity.log</span>
                    </div>

                    <div className="flex-1 overflow-y-auto space-y-1.5 scrollbar-thin scrollbar-thumb-white/10">
                        <AnimatePresence mode="popLayout">
                            {logs.map((log, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className="text-foreground/80"
                                >
                                    <span className="text-foreground-muted select-none mr-2">$</span>
                                    {log}
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        <div className="h-4" /> {/* Spacer */}
                    </div>
                </div>
            </div>
        </div>
    );
};
