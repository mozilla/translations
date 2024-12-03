/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for non-pivot translation requests
 * that contain only plain text without HTML tags.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "en",
    sourceText: "Hola mundo",
    expectedText: "Hello world",
  },
  {
    sourceLanguage: "en",
    targetLanguage: "es",
    sourceText: "Hello world",
    expectedText: "Hola mundo",
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "en",
    sourceText: "Bonjour le monde",
    expectedText: "Hello world",
  },
  {
    sourceLanguage: "en",
    targetLanguage: "fr",
    sourceText: "Hello world",
    expectedText: "Bonjour le monde",
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "en",
    sourceText: "你好，世界",
    expectedText: "Hello, the world.",
  },
];

describe("Plain-Text Non-Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
