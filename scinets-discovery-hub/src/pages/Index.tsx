import { NetworkBackground } from "@/components/NetworkBackground";
import { Header } from "@/components/Header";
import { HeroSection } from "@/components/HeroSection";
import { AgentsSection } from "@/components/AgentsSection";
import { FeaturesSection } from "@/components/FeaturesSection";
import { FooterSection } from "@/components/FooterSection";
import { useEffect } from "react";
import { useLocation } from "react-router-dom";

const Index = () => {
  const location = useLocation();

  useEffect(() => {
    if (location.hash) {
      const element = document.getElementById(location.hash.replace("#", ""));
      if (element) {
        setTimeout(() => {
          element.scrollIntoView({ behavior: "smooth" });
        }, 100);
      }
    }
  }, [location]);

  return (
    <div className="relative min-h-screen bg-background overflow-hidden">
      {/* Animated network background */}
      <NetworkBackground />

      {/* Subtle grid overlay */}
      <div className="fixed inset-0 network-grid pointer-events-none opacity-40" />

      {/* Content */}
      <div className="relative z-10">
        <Header />
        <main>
          <HeroSection />
          <AgentsSection />
          <FeaturesSection />
        </main>
        <FooterSection />
      </div>
    </div>
  );
};

export default Index;
