/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'chat-bg': '#ffffff',
        'message-bg': '#f0f0f0',
        'user-message': '#1a75ff',
      }
    },
  },
  plugins: [],
}

