// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import storybook from "eslint-plugin-storybook";

import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config({ ignores: ['dist'] }, {
  extends: ["plugin:jsx-a11y/recommended"],
  files: ['**/*.{ts,tsx}'],
  languageOptions: {
    ecmaVersion: 2020,
    globals: globals.browser,
  },
  plugins: ["jsx-a11y"],
  "rules": {
  "jsx-a11y/anchor-is-valid": "off"
},
}, storybook.configs["flat/recommended"]);
