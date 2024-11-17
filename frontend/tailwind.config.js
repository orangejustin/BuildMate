/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'chat-bg': '#000000',
        'message-bg': '#262626',
        'user-message': '#4CAF50',
      }
    },
  },
  plugins: [],
}

