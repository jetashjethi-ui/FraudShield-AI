import type React from 'react'
import type { Metadata, Viewport } from 'next'

// The FraudShield UI is the static page at public/index.html (served at "/").
// The App Router only hosts the /api/backend/* server proxy handlers.
export const metadata: Metadata = {
  title: 'FraudShield AI — Risk Scoring Dashboard',
  description: 'Research and demo dashboard for the FraudShield FastAPI scoring service.',
}

export const viewport: Viewport = {
  themeColor: '#050505',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
