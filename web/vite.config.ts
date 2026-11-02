import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  preview: {
    port: 4173,
    // Allow Cloudflare quick-tunnel hosts (the subdomain changes per run).
    allowedHosts: ['.trycloudflare.com'],
  },
});
