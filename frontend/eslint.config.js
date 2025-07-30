// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import storybook from "eslint-plugin-storybook";
import jsxA11y from 'eslint-plugin-jsx-a11y';

import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  // Global ignores - these apply to all configurations
  { 
    ignores: [
      'dist/**', 
      'storybook-static/**', 
      'build/**', 
      'node_modules/**',
      '*.config.js',
      '*.config.ts',
      'vite.config.*',
      '.storybook/**'
    ] 
  },
  
  // Base JavaScript configuration
  js.configs.recommended,
  
  // TypeScript configuration
  ...tseslint.configs.recommended,
  
  // Main application files
  {
    files: ['src/**/*.{ts,tsx,js,jsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'jsx-a11y': jsxA11y,
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      // TypeScript rules - more lenient for now
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': 'warn',
      
      // Accessibility rules - relaxed for development
      ...jsxA11y.configs.recommended.rules,
      'jsx-a11y/anchor-is-valid': 'off',
      'jsx-a11y/click-events-have-key-events': 'warn',
      'jsx-a11y/no-static-element-interactions': 'warn',
      'jsx-a11y/no-noninteractive-element-interactions': 'warn',
      
      // React hooks
      ...reactHooks.configs.recommended.rules,
      
      // React refresh
      'react-refresh/only-export-components': 'warn',
      
      // General rules
      'no-irregular-whitespace': 'error',
    },
  },
  
  // Storybook-specific configuration
  {
    files: ['**/*.stories.@(js|jsx|ts|tsx|mdx)', '.storybook/**/*.@(js|ts)'],
    plugins: {
      storybook: storybook,
    },
    rules: {
      ...(storybook.configs.recommended?.rules || {}),
      '@typescript-eslint/no-explicit-any': 'off', // Allow any in storybook files
    },
  }
);
