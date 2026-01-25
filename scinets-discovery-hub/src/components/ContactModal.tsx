
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Send, Loader2, CheckCircle2 } from "lucide-react";
import { cn } from "../lib/utils.ts";

interface ContactModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const ContactModal = ({ isOpen, onClose }: ContactModalProps) => {
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [formData, setFormData] = useState({
        name: "",
        email: "",
        message: "",
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8005";
            const response = await fetch(`${API_URL}/api/contact`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });

            if (!response.ok) throw new Error("Failed to send message");

            setSuccess(true);
            setTimeout(() => {
                setSuccess(false);
                setFormData({ name: "", email: "", message: "" });
                onClose();
            }, 2000);
        } catch (err) {
            setError("Something went wrong. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-[60] bg-black/50 backdrop-blur-sm flex items-center justify-center p-4"
                        onClick={onClose}
                    >
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="w-full max-w-md relative"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="bg-background border border-border rounded-xl shadow-2xl relative overflow-hidden">
                                {/* Close Button */}
                                <button
                                    onClick={onClose}
                                    className="absolute right-4 top-4 p-2 text-foreground-muted hover:text-foreground transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>

                                <div className="p-6">
                                    <h2 className="text-xl font-semibold mb-1">Contact Us</h2>
                                    <p className="text-sm text-foreground-muted mb-6">
                                        We'd love to hear from you.
                                    </p>

                                    {success ? (
                                        <div className="flex flex-col items-center justify-center py-8 text-center animate-in fade-in zoom-in duration-300">
                                            <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center mb-4">
                                                <CheckCircle2 className="w-6 h-6 text-emerald-500" />
                                            </div>
                                            <h3 className="text-lg font-medium text-foreground">Message Sent!</h3>
                                            <p className="text-sm text-foreground-muted">
                                                Thanks for reaching out. We'll be in touch soon.
                                            </p>
                                        </div>
                                    ) : (
                                        <form onSubmit={handleSubmit} className="space-y-4">
                                            <div className="space-y-2">
                                                <label className="text-xs font-medium text-foreground-muted uppercase tracking-wider">
                                                    Name
                                                </label>
                                                <input
                                                    required
                                                    type="text"
                                                    value={formData.name}
                                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all text-sm"
                                                    placeholder="Your name"
                                                />
                                            </div>

                                            <div className="space-y-2">
                                                <label className="text-xs font-medium text-foreground-muted uppercase tracking-wider">
                                                    Email
                                                </label>
                                                <input
                                                    required
                                                    type="email"
                                                    value={formData.email}
                                                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all text-sm"
                                                    placeholder="you@example.com"
                                                />
                                            </div>

                                            <div className="space-y-2">
                                                <label className="text-xs font-medium text-foreground-muted uppercase tracking-wider">
                                                    Message
                                                </label>
                                                <textarea
                                                    required
                                                    rows={4}
                                                    value={formData.message}
                                                    onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                                                    className="w-full px-3 py-2 rounded-lg bg-secondary/50 border border-border focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all text-sm resize-none"
                                                    placeholder="How can we help?"
                                                />
                                            </div>

                                            {error && (
                                                <p className="text-xs text-red-500">{error}</p>
                                            )}

                                            <button
                                                type="submit"
                                                disabled={loading}
                                                className={cn(
                                                    "w-full flex items-center justify-center gap-2 py-2.5 rounded-lg",
                                                    "bg-foreground text-background font-medium text-sm",
                                                    "hover:bg-foreground/90 transition-colors",
                                                    loading && "opacity-70 cursor-not-allowed"
                                                )}
                                            >
                                                {loading ? (
                                                    <Loader2 className="w-4 h-4 animate-spin" />
                                                ) : (
                                                    <>
                                                        Send Message
                                                        <Send className="w-3.5 h-3.5" />
                                                    </>
                                                )}
                                            </button>
                                        </form>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};
