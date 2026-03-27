/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        /* Light professional theme: white, gray, blue (per spec) */
        'bg-base':   '#f8fafc',
        'bg-card':   '#ffffff',
        'bg-card2':  '#f1f5f9',
        'bg-card3':  '#e2e8f0',
        accent:      '#2563eb',
        'accent-2':  '#64748b',
        danger:      '#dc2626',
        warning:     '#d97706',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4,0,0.6,1) infinite',
        'scan':       'scan 4s linear infinite',
        'slideIn':    'slideIn 0.3s ease',
        'toastIn':    'toastIn 0.35s cubic-bezier(0.34,1.56,0.64,1)',
      },
      keyframes: {
        scan: {
          '0%':   { top: '-2px' },
          '100%': { top: '100%' },
        },
        slideIn: {
          from: { opacity: '0', transform: 'translateX(-10px)' },
          to:   { opacity: '1', transform: 'translateX(0)' },
        },
        toastIn: {
          from: { opacity: '0', transform: 'translateX(40px) scale(0.9)' },
          to:   { opacity: '1', transform: 'translateX(0) scale(1)' },
        },
      },
    },
  },
  plugins: [],
}
