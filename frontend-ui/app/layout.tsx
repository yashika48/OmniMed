import './globals.css';

export const metadata = {
  title: 'OmniMed',
  description: 'OmniMed Diagnostic Analysis Console',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}