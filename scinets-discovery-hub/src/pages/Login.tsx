import { useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { NetworkBackground } from "@/components/NetworkBackground";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Login:", { email, password });
  };

  return (
    <div className="relative min-h-screen bg-background overflow-hidden">
      <NetworkBackground />
      <div className="fixed inset-0 network-grid pointer-events-none opacity-40" />

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-sm"
        >
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 mb-8">
            <div className="w-2 h-2 rounded-full bg-foreground" />
            <span className="text-sm font-medium tracking-tight text-foreground">
              SciNets
            </span>
          </Link>

          <h1 className="text-xl font-medium text-foreground mb-2">
            Welcome back
          </h1>
          <p className="text-sm text-foreground-muted mb-8">
            Sign in to continue your research
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email" className="text-sm text-foreground-muted">
                Email
              </Label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="bg-background border-border/50 focus:border-foreground/50"
                placeholder="you@example.com"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-sm text-foreground-muted">
                Password
              </Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-background border-border/50 focus:border-foreground/50"
                placeholder="••••••••"
              />
            </div>

            <Button
              type="submit"
              className="w-full bg-foreground text-background hover:bg-foreground/90"
            >
              Sign in
            </Button>
          </form>

          <p className="mt-6 text-center text-sm text-foreground-muted">
            Don't have an account?{" "}
            <Link to="/login" className="text-foreground hover:underline">
              Sign up
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default Login;
