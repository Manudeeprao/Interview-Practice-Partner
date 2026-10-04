import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The app talks to the Flask backend directly at VITE_API_URL.
// As an alternative (e.g. to avoid CORS issues), you can point the client
// at this dev-server proxy instead: set VITE_API_URL=/api and requests to
// /api/* are forwarded to the Flask backend on localhost:5000.
// See README.md for details.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
