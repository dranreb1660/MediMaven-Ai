/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    screens: {
      'xs': '480px',
      'sm': '640px', 
      'md': '768px',
      'lg': '1024px',
      'xl': '1280px',
      '2xl': '1536px',
    },
    extend: {
      colors: {
        mm: {
          accent: {
            DEFAULT: '#F36D5E',
            50: '#FFF0EE',
            100: '#FEDCD7',
            200: '#FDB8AF',
            300: '#FB9487',
            400: '#FA6F5F',
            500: '#F36D5E', // DEFAULT
            600: '#D95849',
            700: '#C64837',
            800: '#A6392C',
            900: '#862C23',
          },
          blue: {
            DEFAULT: '#B0CBE1',
            50: '#F3F8FC',
            100: '#E1EEF7',
            200: '#C7DFEF',
            300: '#B0CBE1', // DEFAULT
            400: '#97B7D3',
            500: '#80A2C6',
            600: '#6482A2',
            700: '#4C617B',
            800: '#344254',
            900: '#202A36',
          },
          bg: {
            DEFAULT: '#F6F4F0',
            dark: '#121212',
          },
          bubble: {
            DEFAULT: '#B0CBE1',
            user: '#F36D5E',
            assistant: '#FFFFFF',
            assistantDark: '#2A2A2A',
          },
        },
      },
    },
  },
  plugins: [
  require('tailwindcss-safe-area'),
  require('@tailwindcss/typography'),
],

};
