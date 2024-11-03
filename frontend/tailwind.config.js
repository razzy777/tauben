/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
  safelist: [
    'bg-blue-500',
    'bg-blue-600',
    'bg-blue-700',
    'bg-gray-500',
    'bg-gray-600',
    'bg-gray-700',
    'hover:bg-blue-600',
    'hover:bg-blue-700',
    'hover:bg-gray-600',
    'focus:ring-blue-400',
    'focus:ring-gray-400',
  ]
}