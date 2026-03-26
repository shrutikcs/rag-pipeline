import {
  Show,
  SignInButton,
  SignUpButton,
  UserButton,
} from "@clerk/nextjs";
import { Button } from "@/components/ui/button";

const Navigation = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60">
      <div className="container flex h-14 max-w-screen-2xl items-center justify-between px-4 mx-auto">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center font-bold text-primary-foreground shadow-sm shadow-primary/20">
            R
          </div>
          <span className="font-bold hidden sm:inline-block tracking-tight">
            RAG Chat-Bot
          </span>
        </div>
        <div className="flex items-center gap-4">
          <Show when="signed-out">
            <div className="flex items-center gap-2">
              <SignInButton mode="modal">
                <Button variant="ghost" size="sm" className="font-medium">
                  Sign In
                </Button>
              </SignInButton>
              <SignUpButton mode="modal">
                <Button size="sm" className="font-semibold shadow-md shadow-primary/20">
                  Get Started
                </Button>
              </SignUpButton>
            </div>
          </Show>
          <Show when="signed-in">
            <UserButton
              appearance={{
                elements: {
                  userButtonAvatarBox: "w-9 h-9 border-2 border-primary/20 hover:border-primary/40 transition-colors",
                },
              }}
            />
          </Show>
        </div>
      </div>
    </header>
  );
};

export default Navigation;