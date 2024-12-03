import js from "@eslint/js";
import prettierPlugin from "eslint-plugin-prettier";
import prettierConfig from "eslint-config-prettier";

export default [
  {
    ignores: ["generated/**/*"],
  },
  js.configs.recommended,
  prettierConfig,
  {
    plugins: {
      prettier: prettierPlugin,
    },
    languageOptions: {
      globals: {
        console: "readonly",
      },
    },
    rules: {
      curly: ["error", "all"],
      indent: [
        "error",
        2,
        {
          SwitchCase: 1,
        },
      ],
      "prefer-const": "error",
      semi: "error",
      quotes: [
        "error",
        "double",
        {
          // Allow single quotes if it avoids escaping double quotes
          avoidEscape: true,
          // Allow backticks for template literals
          allowTemplateLiterals: true,
        },
      ],
      "prettier/prettier": "error",
    },
  },
];
