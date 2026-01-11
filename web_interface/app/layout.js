
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "CryptoHunter AI Dashboard",
  description: "Advanced AI-Powered Crypto Screener",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-background text-foreground min-h-screen antialiased`}>
        {children}
      </body>
    </html>
  );
}
