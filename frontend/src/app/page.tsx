import { Show } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowRight, Bot, Sparkles, Zap } from "lucide-react";

export default function Home() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center relative overflow-hidden bg-background">
      {/* Background Decorative Elements */}
      <div className="absolute top-0 -left-4 w-72 h-72 bg-primary/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-0 -right-4 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />

      <main className="relative z-10 w-full max-w-5xl px-6 py-20 flex flex-col items-center text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 text-primary text-sm font-medium mb-8 animate-in fade-in slide-in-from-bottom-3 duration-1000">
          <Sparkles className="w-4 h-4" />
          <span>Next-Generation RAG Technology</span>
        </div>

        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 bg-linear-to-b from-foreground to-foreground/70 bg-clip-text text-transparent">
          The Chat Bot That <br />
          <span className="text-primary italic">Doesn't Suck.</span>
        </h1>

        <p className="max-w-2xl text-lg md:text-xl text-muted-foreground mb-10 mx-auto">
          Built with advanced Retrieval-Augmented Generation to provide
          accurate, context-aware answers from your documentation and beyond.
        </p>

        <div className="flex flex-col sm:flex-row items-center gap-4">
          <Link href="/chat">
            <Button size="lg" className="h-12 px-8 text-lg gap-2">
              Start Chatting <ArrowRight className="w-5 h-5" />
            </Button>
          </Link>

          <Show when="signed-out">
            <Link href="/chat">
              <Button size="lg" variant="outline" className="h-12 px-8 text-lg">
                View Demo
              </Button>
            </Link>
          </Show>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-24 w-full">
          <FeatureCard
            icon={<Zap className="w-6 h-6 text-primary" />}
            title="Ultra-Fast"
            description="Responses delivered in milliseconds thanks to optimized vector search."
          />
          <FeatureCard
            icon={<Bot className="w-6 h-6 text-primary" />}
            title="Context Aware"
            description="The AI understands your specific data and maintains deep context."
          />
          <FeatureCard
            icon={<Sparkles className="w-6 h-6 text-primary" />}
            title="Premium Design"
            description="A sleek, modern interface designed for focus and productivity."
          />
        </div>
      </main>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="group p-6 rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm transition-all hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5 text-left">
      <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </div>
  );
}
