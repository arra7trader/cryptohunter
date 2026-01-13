import { Inter } from "next/font/google";
import "./globals.css";
import WalletContextProvider from "@/components/WalletContextProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "CryptoHunter AI Dashboard",
  description: "Advanced AI-Powered Crypto Screener",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.className} bg-background text-foreground min-h-screen antialiased`} suppressHydrationWarning>
        <WalletContextProvider>
          {children}
        </WalletContextProvider>
      </body>
    </html>
  );
}
