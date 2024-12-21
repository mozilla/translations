/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for non-pivot translation requests
 * that contain plain text with multiple sentences, without HTML tags.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "en",
    sourceText: "Hola mundo. ¿Cómo estás? Espero que todo esté bien.",
    expectedText: "Hello world. How are you? I hope everything's okay.",
  },
  {
    sourceLanguage: "en",
    targetLanguage: "es",
    sourceText: "Hello world. How are you? I hope everything is well.",
    expectedText: "Hola mundo. Cómo estás? Espero que todo esté bien.",
  },
  {
    sourceLanguage: "en",
    targetLanguage: "zh",
    sourceText: "Hello world. How are you? I hope everything is well.",
    expectedText:
      // This is temporarily using a Spanish model until we have a trained enzh model.
      // The relevance of the assertion is that whitespace has been omitted between sentences.
      "Hola mundo.Cómo estás?Espero que todo esté bien.",
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "en",
    sourceText: "Bonjour le monde. Comment ça va? J'espère que tout va bien.",
    expectedText: "Hello world. How are you doing? I hope all is well.",
  },
  {
    sourceLanguage: "en",
    targetLanguage: "fr",
    sourceText: "Hello world. How are you? I hope everything is well.",
    expectedText:
      "Bonjour au monde. Comment vas-tu ? J'espère que tout va bien.",
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText: "你好，世界。你好吗？我希望一切都好。",
    expectedText:
      "Hello, the world. Are you, how well? I hope everything is okay.",
  },
];

describe("Plain-Text Multi-Sentence Non-Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
