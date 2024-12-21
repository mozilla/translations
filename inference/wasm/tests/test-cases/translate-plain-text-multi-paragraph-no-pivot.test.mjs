/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for non-pivot translation requests
 * that contain plain text with multiple paragraphs, without HTML tags.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "en",
    sourceText:
      "Hola mundo. ¿Cómo estás?\n\nEspero que todo esté bien. Nos vemos pronto.",
    expectedText:
      "Hello world. How are you?\n\nI hope everything's okay. See you soon.",
  },
  {
    sourceLanguage: "en",
    targetLanguage: "es",
    sourceText:
      "Hello world. How are you?\n\nI hope everything is well. See you soon.",
    expectedText:
      "Hola mundo. Cómo estás?\n\nEspero que todo esté bien. Hasta pronto.",
  },
  {
    sourceLanguage: "en",
    targetLanguage: "zh",
    sourceText:
      "Hello world. How are you?\n\nI hope everything is well. See you soon.",
    expectedText:
      // This is temporarily using a Spanish model until we have a trained enzh model.
      // The relevance of the assertion is that whitespace has been omitted between sentences.
      "Hola mundo.Cómo estás?\n\nEspero que todo esté bien.Hasta pronto.",
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "en",
    sourceText:
      "Bonjour le monde. Comment ça va?\n\nJ'espère que tout va bien. À bientôt.",
    expectedText:
      "Hello world. How are you doing?\n\nI hope all is well. See you soon.",
  },
  {
    sourceLanguage: "en",
    targetLanguage: "fr",
    sourceText:
      "Hello world. How are you?\n\nI hope everything is well. See you soon.",
    expectedText:
      "Bonjour au monde. Comment vas-tu ?\n\nJ'espère que tout va bien. À bientôt.",
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText: "你好，世界。你好吗？\n\n我希望一切都好。再见。",
    expectedText:
      "Hello, the world. Are you, how well?\n\nI hope everything is okay. Farewell to me.",
  },
];

describe("Plain-Text Multi-Paragraph Non-Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
