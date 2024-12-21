/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import { describe } from "vitest";
import { runTranslationTest } from "./test-cases/shared.mjs";

/**
 * This file tests the WASM bindings for pivot translation requests
 * that contain only plain text without HTML tags.
 */
const testCases = [
  {
    sourceLanguage: "es",
    targetLanguage: "fr",
    sourceText: "El perro azul.",
    expectedText: "Le chien bleu.",
  },
  {
    sourceLanguage: "es",
    targetLanguage: "fr",
    sourceText: "El perro azul",
    expectedText: "Le chien bleu",
  },
  {
    sourceLanguage: "fr",
    targetLanguage: "es",
    sourceText: "Le chien bleu.",
    expectedText: "El perro azul.",
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "es",
    sourceText: "蓝色的狗。",
    expectedText: "El perro azul.",
  },
  {
    sourceLanguage: "zh",
    targetLanguage: "fr",
    sourceText: "蓝色的狗。",
    expectedText: "Le chien bleu.",
  },
];

describe("Plain-Text Pivot Translations", () => {
  testCases.forEach(runTranslationTest);
});
