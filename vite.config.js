import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';

// No client-side routing here — disable Vite's SPA history fallback so a
// missing static asset (e.g. a model file that isn't under public/) 404s
// for real instead of silently resolving to index.html's HTML as a "200".
export default defineConfig({
  appType: 'mpa',
  plugins: [tailwindcss()],
});
