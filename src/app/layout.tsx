import { cn } from "@/lib/utils";
import {
  ClerkProvider
} from "@clerk/nextjs";
import type { Metadata } from "next";
import { Bricolage_Grotesque } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/navigation";


const briGro = Bricolage_Grotesque({
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAG Chat-Bot",
  description: "A Chat Bot that doesn't suck.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={cn("dark h-full", "antialiased", briGro.className)}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col">
        <ClerkProvider>
          <Navigation/>
          {children}
        </ClerkProvider>
      </body>
    </html>
  );
}
