/** Design tokens taken from the Stitch design system (see docs/stitch). */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0b0e16', surface: '#121723', card: '#161c2b', line: '#232b3f',
        primary: '#6366f1', secondary: '#06b6d4', tertiary: '#14b8a6',
        muted: '#8b94ab',
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        sans: ['Geist', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      boxShadow: { glow: '0 0 0 1px rgba(99,102,241,.35), 0 8px 30px -10px rgba(99,102,241,.45)' },
    },
  },
  plugins: [],
}
