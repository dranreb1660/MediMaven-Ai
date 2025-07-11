/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',           // <-- enable dark-mode
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        mm: {
          accent: '#F36D5E',
          accentDark: '#D95849',
          accentDarker: '#C64837',
          bubble: '#B0CBE1',
          bg: '#F6F4F0',
        },
      },
    },
  },
  plugins: [require('tailwindcss-safe-area')],
};
